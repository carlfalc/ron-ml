"""
RON AI - Machine Learning Trading Intelligence Server
GAINEDGE Proprietary Trading Model

This server trains and serves an XGBoost model that predicts
trade outcomes based on market conditions. It learns from all
GAINEDGE users' signal outcomes to improve over time.

Deploy to: Render.com ($7/month)
"""

import os
import json
import logging
import uuid
import lzma
import struct
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests

# ─── Configuration ───────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["https://gainedge.ai", "https://preview--conviction-edge-pro.lovable.app"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ron-ml")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://ecsztqtyttnqdnsphxip.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
MODEL_PATH = "ron_model.joblib"
FEATURE_NAMES_PATH = "ron_features.json"
MIN_TRAINING_SAMPLES = 50  # Minimum signals needed to train

# ─── Supabase Helper ─────────────────────────────────────────
def supabase_query(table: str, params: dict = None) -> list:
    """Query Supabase REST API directly."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    resp = requests.get(url, headers=headers, params=params or {})
    if resp.status_code == 200:
        return resp.json()
    logger.error(f"Supabase query failed: {resp.status_code} {resp.text}")
    return []


def supabase_insert(table: str, data: dict) -> bool:
    """Insert into Supabase."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    resp = requests.post(url, headers=headers, json=data)
    return resp.status_code in (200, 201)


# ─── Feature Engineering ─────────────────────────────────────
FEATURE_COLUMNS = [
    "adx_at_entry",
    "rsi_at_entry",
    "stoch_rsi_at_entry",
    "macd_bullish",          # 1 = bullish, 0 = neutral, -1 = bearish
    "confidence",
    "is_buy",                # 1 = buy, 0 = sell
    "session_asian",         # one-hot encoded sessions
    "session_london",
    "session_ny",
    "session_overlap",
    "hour_sin",              # cyclical encoding of hour
    "hour_cos",
    "day_sin",               # cyclical encoding of day of week
    "day_cos",
    "pattern_active",        # 1 = pattern detected, 0 = none
    "pattern_double_top",
    "pattern_double_bottom",
    "pattern_head_shoulders",
    "pattern_bull_flag",
    "pattern_bear_flag",
    "pattern_triangle",
    "instrument_xauusd",     # one-hot instruments
    "instrument_us30",
    "instrument_nas100",
    "instrument_nzdusd",
    "instrument_audusd",
    "instrument_eurusd",
    "instrument_gbpusd",
    "instrument_usdjpy",
]


def encode_features(row: dict) -> dict:
    """Convert a signal_outcomes row into ML features."""
    features = {}

    # Numeric indicators (with defaults)
    features["adx_at_entry"] = float(row.get("adx_at_entry") or 20)
    features["rsi_at_entry"] = float(row.get("rsi_at_entry") or 50)
    features["stoch_rsi_at_entry"] = float(row.get("stoch_rsi_at_entry") or 50)
    features["confidence"] = int(row.get("confidence") or 5)

    # MACD status
    macd = str(row.get("macd_status") or "Neutral").lower()
    features["macd_bullish"] = 1 if macd == "bullish" else (-1 if macd == "bearish" else 0)

    # Direction
    direction = str(row.get("direction") or "BUY").upper()
    features["is_buy"] = 1 if direction == "BUY" else 0

    # Session encoding
    session = str(row.get("session") or "").lower()
    features["session_asian"] = 1 if "asian" in session else 0
    features["session_london"] = 1 if "london" in session else 0
    features["session_ny"] = 1 if "ny" in session or "new_york" in session else 0
    features["session_overlap"] = 1 if "overlap" in session else 0

    # Cyclical time encoding (preserves the circular nature of hours/days)
    hour = int(row.get("hour_utc") or 12)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    day = int(row.get("day_of_week") or 3)
    features["day_sin"] = np.sin(2 * np.pi * day / 7)
    features["day_cos"] = np.cos(2 * np.pi * day / 7)

    # Pattern encoding
    pattern = str(row.get("pattern_active") or "").lower()
    features["pattern_active"] = 1 if pattern and pattern != "none" else 0
    features["pattern_double_top"] = 1 if "double top" in pattern else 0
    features["pattern_double_bottom"] = 1 if "double bottom" in pattern else 0
    features["pattern_head_shoulders"] = 1 if "head" in pattern else 0
    features["pattern_bull_flag"] = 1 if "bull flag" in pattern else 0
    features["pattern_bear_flag"] = 1 if "bear flag" in pattern else 0
    features["pattern_triangle"] = 1 if "triangle" in pattern else 0

    # Instrument encoding
    symbol = str(row.get("symbol") or "").upper()
    features["instrument_xauusd"] = 1 if "XAU" in symbol else 0
    features["instrument_us30"] = 1 if "US30" in symbol else 0
    features["instrument_nas100"] = 1 if "NAS" in symbol or "NDX" in symbol else 0
    features["instrument_nzdusd"] = 1 if "NZD" in symbol else 0
    features["instrument_audusd"] = 1 if "AUD" in symbol else 0
    features["instrument_eurusd"] = 1 if "EUR" in symbol else 0
    features["instrument_gbpusd"] = 1 if "GBP" in symbol else 0
    features["instrument_usdjpy"] = 1 if "JPY" in symbol else 0

    return features


# ─── Model Training ──────────────────────────────────────────
def train_model() -> dict:
    """
    Train the XGBoost model on all signal_outcomes data.
    Returns training metrics.
    """
    logger.info("Starting RON model training...")

    # Fetch all resolved signal outcomes from ALL users
    rows = supabase_query("signal_outcomes", {
        "select": "*",
        "result": "neq.PENDING",
        "order": "created_at.desc",
        "limit": "10000"
    })

    if len(rows) < MIN_TRAINING_SAMPLES:
        return {
            "status": "insufficient_data",
            "samples": len(rows),
            "required": MIN_TRAINING_SAMPLES,
            "message": f"Need {MIN_TRAINING_SAMPLES} resolved signals to train. Currently have {len(rows)}."
        }

    # Encode features and labels
    X_data = []
    y_data = []

    for row in rows:
        result = str(row.get("result", "")).upper()
        if result not in ("WIN", "LOSS"):
            continue  # Skip expired for training — we want clear win/loss

        features = encode_features(row)
        X_data.append([features.get(col, 0) for col in FEATURE_COLUMNS])
        y_data.append(1 if result == "WIN" else 0)

    if len(X_data) < MIN_TRAINING_SAMPLES:
        return {
            "status": "insufficient_win_loss_data",
            "samples": len(X_data),
            "required": MIN_TRAINING_SAMPLES
        }

    X = np.array(X_data)
    y = np.array(y_data)

    # Split: 80% train, 20% test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_accuracy = np.mean(train_preds == y_train)
    test_accuracy = np.mean(test_preds == y_test)

    # Feature importance
    importance = model.feature_importances_
    feature_ranking = sorted(
        zip(FEATURE_COLUMNS, importance),
        key=lambda x: x[1],
        reverse=True
    )

    # Save model
    joblib.dump(model, MODEL_PATH)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    metrics = {
        "status": "trained",
        "total_samples": len(X_data),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_accuracy": round(float(train_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "win_rate_in_data": round(float(np.mean(y)), 4),
        "top_features": [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in feature_ranking[:10]
        ],
        "trained_at": datetime.utcnow().isoformat()
    }

    logger.info(f"Model trained: {metrics}")

    # Store training results in Supabase insights
    supabase_insert("insights", {
        "insight_type": "ron_ml_training",
        "title": "RON ML Model Trained",
        "description": json.dumps(metrics),
        "created_at": datetime.utcnow().isoformat()
    })

    return metrics


def load_model():
    """Load the trained model from disk."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


# ─── Prediction ──────────────────────────────────────────────
def predict_win_probability(market_conditions: dict) -> dict:
    """
    Given current market conditions, predict the probability
    of a trade winning.
    """
    model = load_model()
    if model is None:
        return {
            "probability": 0.5,
            "confidence_label": "No model trained yet",
            "model_available": False
        }

    features = encode_features(market_conditions)
    X = np.array([[features.get(col, 0) for col in FEATURE_COLUMNS]])

    # Get probability of WIN (class 1)
    prob = float(model.predict_proba(X)[0][1])

    # Generate confidence label
    if prob >= 0.75:
        label = "HIGH CONVICTION"
    elif prob >= 0.60:
        label = "MODERATE"
    elif prob >= 0.45:
        label = "LOW"
    else:
        label = "AVOID"

    # Get feature contributions (which factors matter most for THIS prediction)
    feature_values = {col: features.get(col, 0) for col in FEATURE_COLUMNS}

    return {
        "probability": round(prob, 4),
        "confidence_label": label,
        "model_available": True,
        "recommendation": "TAKE TRADE" if prob >= 0.60 else "SKIP",
        "key_factors": feature_values
    }


# ─── API Endpoints ───────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    model = load_model()
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/train", methods=["POST"])
def train():
    """
    Trigger model training. Called weekly by a cron job
    or manually by admin.
    """
    # Simple API key auth
    api_key = request.headers.get("X-API-Key", "")
    expected_key = os.environ.get("RON_API_KEY", "gainedge-ron-2026")
    if api_key != expected_key:
        return jsonify({"error": "Unauthorized"}), 401

    result = train_model()
    return jsonify(result)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict win probability for a potential trade.

    Expected JSON body:
    {
        "symbol": "XAUUSD",
        "direction": "BUY",
        "adx_at_entry": 28.5,
        "rsi_at_entry": 62.3,
        "stoch_rsi_at_entry": 71.5,
        "macd_status": "Bullish",
        "confidence": 7,
        "session": "london",
        "hour_utc": 9,
        "day_of_week": 2,
        "pattern_active": "Double Bottom"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    result = predict_win_probability(data)
    return jsonify(result)


@app.route("/predict-v2", methods=["POST"])
def predict_v2():
    """
    PCF (Precision Confluence v2) enhanced prediction.

    Extends /predict with explicit PCF filter flags that adjust the base ML
    probability before deciding whether RON should EXECUTE or HOLD.

    PCF-specific fields (bool):
        ema_stack_aligned  EMA8 > EMA21 and price correct side of EMA50
        htf_aligned        1H EMA9/21 confirm the 15m direction
        adx_above_20       ADX(14) >= 20 at entry
        rsi_in_zone        RSI in PCF zone (BUY 45-72 / SELL 28-55)
        in_session         London or NY session active

    Optional:
        min_probability    float threshold for EXECUTE action (default 0.65)
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    base_result = predict_win_probability(data)
    base_prob = base_result["probability"]

    adx_val = float(data.get("adx_at_entry", 0))
    pcf_filters = {
        "ema_stack_aligned": bool(data.get("ema_stack_aligned", False)),
        "htf_aligned":       bool(data.get("htf_aligned", False)),
        "adx_above_20":      bool(data.get("adx_above_20", adx_val >= 20)),
        "rsi_in_zone":       bool(data.get("rsi_in_zone", False)),
        "in_session":        bool(data.get("in_session", False)),
    }

    # Each passing filter boosts probability; each failing filter penalises
    FILTER_WEIGHTS = {
        "ema_stack_aligned": (0.05, -0.05),
        "htf_aligned":       (0.05, -0.05),
        "adx_above_20":      (0.03, -0.03),
        "rsi_in_zone":       (0.03, -0.03),
        "in_session":        (0.02, -0.02),
    }

    adjustment = 0.0
    filter_detail = {}
    for fname, passes in pcf_filters.items():
        boost, penalty = FILTER_WEIGHTS[fname]
        delta = boost if passes else penalty
        adjustment += delta
        filter_detail[fname] = {"pass": passes, "adjustment": round(delta, 4)}

    adjusted_prob = round(min(max(base_prob + adjustment, 0.0), 1.0), 4)
    filters_all_pass = all(pcf_filters.values())
    min_prob = float(data.get("min_probability", 0.65))
    ron_action = "EXECUTE" if (adjusted_prob >= min_prob and filters_all_pass) else "HOLD"

    if adjusted_prob >= 0.75:
        label = "HIGH CONVICTION"
    elif adjusted_prob >= 0.65:
        label = "MODERATE"
    elif adjusted_prob >= 0.50:
        label = "LOW"
    else:
        label = "AVOID"

    return jsonify({
        "base_probability":     base_prob,
        "adjusted_probability": adjusted_prob,
        "pcf_filters":          filter_detail,
        "filters_all_pass":     filters_all_pass,
        "ron_action":           ron_action,
        "confidence_label":     label,
        "min_probability":      min_prob,
        "model_available":      base_result.get("model_available", False),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# DLO + SQUEEZE + HEIKIN ASHI ENGINE  (predict-v3)
# Replicates the Pine Script DLO+Squeeze Combined v3 + EMA 12/69 server-side.
# No session rules applied here — caller controls session toggling.
# ═══════════════════════════════════════════════════════════════════════════════

def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing — matches Pine Script ta.rma (alpha = 1/period)."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _ema_s(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _sma_s(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Returns (plus_di, minus_di, adx) using Wilder smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move.fillna(0),   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move.fillna(0), 0.0)

    tr_s       = _rma(tr.fillna(0), period)
    plus_di    = 100 * _rma(pd.Series(plus_dm,  index=high.index), period) / tr_s
    minus_di   = 100 * _rma(pd.Series(minus_dm, index=high.index), period) / tr_s
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = _rma(dx, period)
    return plus_di, minus_di, adx


def _logistic_prob(series: pd.Series, lookback: int, slope: float, smooth: int) -> pd.Series:
    mean = _sma_s(series, lookback)
    z    = (series - mean) * slope
    raw  = 1.0 / (1.0 + np.exp(-z.clip(-20, 20)))
    return _ema_s(raw, smooth)


def _tanh_series(x: pd.Series) -> pd.Series:
    e2x = np.exp((2 * x).clip(-40, 40))
    return (e2x - 1) / (e2x + 1)


def _linreg_rolling(series: pd.Series, length: int) -> pd.Series:
    """Rolling linear regression value at the last point of each window."""
    vals   = series.values.astype(float)
    result = np.full(len(vals), np.nan)
    x      = np.arange(length, dtype=float)
    xm     = x.mean()
    xvar   = np.dot(x - xm, x - xm)
    for i in range(length - 1, len(vals)):
        y = vals[i - length + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        ym    = y.mean()
        slope = np.dot(x - xm, y - ym) / xvar
        result[i] = ym + slope * (length - 1 - xm)
    return pd.Series(result, index=series.index)


def _calc_dlo(high, low, close,
              di_len=14, mean_lb=360, slope=0.18,
              smooth_len=3, osc_scale=2.5, osc_smooth_len=7):
    plus_di, minus_di, adx = _dmi(high, low, close, di_len)
    prob_p   = _logistic_prob(plus_di,  mean_lb, slope, smooth_len)
    prob_m   = _logistic_prob(minus_di, mean_lb, slope, smooth_len)
    prob_adx = _logistic_prob(adx,      mean_lb, slope, smooth_len)
    strength = (prob_p - prob_m) * prob_adx * osc_scale
    dlo      = _ema_s(_tanh_series(strength), smooth_len)
    dlo_sm   = _ema_s(dlo, osc_smooth_len)
    return dlo, dlo_sm


def _calc_squeeze(close, high, low,
                  bb_len=20, bb_mult=2.0, kc_len=20, kc_mult=1.5):
    basis    = _sma_s(close, bb_len)
    bb_dev   = bb_mult * close.rolling(bb_len, min_periods=bb_len).std(ddof=0)
    upper_bb = basis + bb_dev
    lower_bb = basis - bb_dev

    prev_c = close.shift(1)
    tr     = pd.concat([high - low,
                        (high - prev_c).abs(),
                        (low  - prev_c).abs()], axis=1).max(axis=1)
    ma_kc    = _sma_s(close, kc_len)
    rng_ma   = _sma_s(tr, kc_len)
    upper_kc = ma_kc + rng_ma * kc_mult
    lower_kc = ma_kc - rng_ma * kc_mult

    sqz_on  = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)

    mid     = (high.rolling(kc_len).max() + low.rolling(kc_len).min()) / 2
    mid     = (mid + _sma_s(close, kc_len)) / 2
    sqz_val = _linreg_rolling(close - mid, kc_len)
    return sqz_on, sqz_off, sqz_val


def _calc_heikin_ashi(open_, high, low, close):
    ha_close_vals = ((open_ + high + low + close) / 4.0).values
    ha_open_vals  = np.zeros(len(open_))
    ha_open_vals[0] = (open_.iloc[0] + close.iloc[0]) / 2.0
    for i in range(1, len(ha_open_vals)):
        ha_open_vals[i] = (ha_open_vals[i - 1] + ha_close_vals[i - 1]) / 2.0
    ha_open  = pd.Series(ha_open_vals,  index=open_.index)
    ha_close = pd.Series(ha_close_vals, index=open_.index)
    ha_high  = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low   = pd.concat([low,  ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close


@app.route("/predict-v3", methods=["POST"])
def predict_v3():
    """
    DLO + Squeeze Momentum + Heikin Ashi + EMA 12/69 signal engine.

    Replicates the DLO+Squeeze Combined v3 Pine Script fully server-side.
    No session filtering applied — RON fires on signal quality alone.
    Sessions are toggled externally (Lovable UI / ron_settings).

    Required body fields:
        bars  — list of [timestamp, open, high, low, close, volume]
                (at least 100 bars; 400+ recommended for full DLO warm-up)

    Optional:
        htf_bars          — 1H OHLCV bars for EMA 69 HTF bias (list, same format)
        min_tier          — "A" | "B" | "C"  (default "B")
        conviction_threshold   — A-tier DLO floor (default 0.25)
        min_b_conviction       — B-tier DLO floor (default 0.15)
        fire_lookback          — bars since squeeze fired (default 10)
        require_squeeze_fire   — bool, A-tier requires squeeze fire (default true)
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    bars = data.get("bars")
    if not bars or len(bars) < 50:
        return jsonify({"error": "Need at least 50 OHLCV bars", "min_bars": 50,
                        "provided": len(bars) if bars else 0}), 400

    # ── Build DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)

    # ── Parameters ───────────────────────────────────────────────────────────
    conviction_a        = float(data.get("conviction_threshold", 0.25))
    conviction_b        = float(data.get("min_b_conviction", 0.15))
    fire_lookback       = int(data.get("fire_lookback", 10))
    min_tier            = str(data.get("min_tier", "B")).upper()
    require_sqz_fire    = bool(data.get("require_squeeze_fire", True))
    mean_lb             = min(360, max(20, n - 30))

    # ── HTF bias (optional 1H bars → EMA 69) ────────────────────────────────
    htf_bias = "NEUTRAL"
    htf_bars = data.get("htf_bars")
    if htf_bars and len(htf_bars) >= 70:
        htf_df = pd.DataFrame(htf_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        htf_df["close"] = pd.to_numeric(htf_df["close"], errors="coerce")
        htf_df = htf_df.sort_values("timestamp").reset_index(drop=True)
        htf_ema69 = _ema_s(htf_df["close"], 69).iloc[-1]
        last_htf  = htf_df["close"].iloc[-1]
        if not np.isnan(htf_ema69):
            htf_bias = "BULL" if last_htf > htf_ema69 else "BEAR"

    # ── Indicators ───────────────────────────────────────────────────────────
    dlo_s, _          = _calc_dlo(df["high"], df["low"], df["close"], mean_lb=mean_lb)
    sqz_on, sqz_off, sqz_val = _calc_squeeze(df["close"], df["high"], df["low"])
    _, _, _, ha_close = _calc_heikin_ashi(df["open"], df["high"], df["low"], df["close"])
    ha_bull_series    = ha_close > df.apply(lambda r: (r["open"] + r["high"] + r["low"] + r["close"]) / 4 * 0, axis=1)
    # Recompute ha_open properly for ha_bull
    ha_open_vals  = np.zeros(n)
    ha_close_vals = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4.0).values
    ha_open_vals[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
    for i in range(1, n):
        ha_open_vals[i] = (ha_open_vals[i - 1] + ha_close_vals[i - 1]) / 2.0
    ha_bull_series = pd.Series(ha_close_vals > ha_open_vals, index=df.index)

    ema12 = _ema_s(df["close"], 12)
    ema69 = _ema_s(df["close"], 69)

    # ── Last-bar values ──────────────────────────────────────────────────────
    cur_dlo       = float(dlo_s.iloc[-1])
    prev_dlo      = float(dlo_s.iloc[-2])
    prev2_dlo     = float(dlo_s.iloc[-3])
    cur_sqz_val   = float(sqz_val.iloc[-1])
    prev_sqz_val  = float(sqz_val.iloc[-2])
    cur_sqz_on    = bool(sqz_on.iloc[-1])
    cur_sqz_off   = bool(sqz_off.iloc[-1])
    cur_ha_bull   = bool(ha_bull_series.iloc[-1])
    prev_ha_bull  = bool(ha_bull_series.iloc[-2])
    cur_ema12     = float(ema12.iloc[-1])
    cur_ema69     = float(ema69.iloc[-1])

    if any(np.isnan(v) for v in [cur_dlo, cur_sqz_val, cur_ema12, cur_ema69]):
        return jsonify({"error": "Insufficient bars for indicator warm-up",
                        "bars_provided": n, "mean_lb_used": mean_lb}), 400

    # ── Derived states ───────────────────────────────────────────────────────
    dlo_rising   = cur_dlo  > prev_dlo  > prev2_dlo
    dlo_falling  = cur_dlo  < prev_dlo  < prev2_dlo
    sqz_accel_up = cur_sqz_val > 0 and cur_sqz_val > prev_sqz_val
    sqz_accel_dn = cur_sqz_val < 0 and cur_sqz_val < prev_sqz_val
    sqz_decel_up = cur_sqz_val > 0 and cur_sqz_val < prev_sqz_val
    sqz_decel_dn = cur_sqz_val < 0 and cur_sqz_val > prev_sqz_val
    ha_transition = cur_ha_bull != prev_ha_bull

    # Squeeze "just fired": currently sqz_off AND sqz_on was true recently
    sqz_just_fired = False
    if cur_sqz_off and n > fire_lookback + 1:
        sqz_just_fired = bool(sqz_on.iloc[-(fire_lookback + 1):-1].any())
    squeeze_state = "ON" if cur_sqz_on else ("FIRED" if sqz_just_fired else "OFF")

    # ── Signal tiers (matching Pine Script logic) ────────────────────────────
    bull_aligned = cur_dlo > 0 and cur_sqz_val > 0
    bear_aligned = cur_dlo < 0 and cur_sqz_val < 0

    sqz_fire_ok = sqz_just_fired if require_sqz_fire else True

    a_bull = bull_aligned and cur_dlo >  conviction_a and sqz_accel_up and sqz_fire_ok
    a_bear = bear_aligned and cur_dlo < -conviction_a and sqz_accel_dn and sqz_fire_ok
    b_bull = bull_aligned and sqz_accel_up and dlo_rising  and cur_dlo >  conviction_b and not a_bull
    b_bear = bear_aligned and sqz_accel_dn and dlo_falling and cur_dlo < -conviction_b and not a_bear
    c_bull = (cur_dlo > 0 and prev_dlo <= 0 and cur_sqz_val > prev_sqz_val) and not a_bull and not b_bull
    c_bear = (cur_dlo < 0 and prev_dlo >= 0 and cur_sqz_val < prev_sqz_val) and not a_bear and not b_bear

    exit_long  = (cur_dlo > 0 and sqz_decel_up) or (cur_dlo < 0 and prev_dlo >= 0)
    exit_short = (cur_dlo < 0 and sqz_decel_dn) or (cur_dlo > 0 and prev_dlo <= 0)

    # ── Resolve direction + tier ─────────────────────────────────────────────
    direction, tier = None, "NONE"
    tier_order = {"A": 0, "B": 1, "C": 2}
    min_tier_rank = tier_order.get(min_tier, 1)

    if a_bull:                         direction, tier = "BUY",  "A"
    elif a_bear:                       direction, tier = "SELL", "A"
    elif b_bull and min_tier_rank >= 1: direction, tier = "BUY",  "B"
    elif b_bear and min_tier_rank >= 1: direction, tier = "SELL", "B"
    elif c_bull and min_tier_rank >= 2: direction, tier = "BUY",  "C"
    elif c_bear and min_tier_rank >= 2: direction, tier = "SELL", "C"

    # HA must agree with signal direction (candle color confirmation)
    ha_confirms = (direction == "BUY" and cur_ha_bull) or (direction == "SELL" and not cur_ha_bull)
    ron_action  = "EXECUTE" if (direction is not None and ha_confirms) else "HOLD"

    label_map   = {"A": "HIGH CONVICTION", "B": "MODERATE", "C": "LOW", "NONE": "AVOID"}

    return jsonify({
        "signal":            direction or "HOLD",
        "ron_action":        ron_action,
        "tier":              tier,
        "confidence_label":  label_map.get(tier, "AVOID"),
        "dlo":               round(cur_dlo, 4),
        "squeeze_state":     squeeze_state,
        "sqz_val":           round(cur_sqz_val, 6),
        "ha_bullish":        cur_ha_bull,
        "ha_transition":     ha_transition,
        "ema12":             round(cur_ema12, 5),
        "ema69":             round(cur_ema69, 5),
        "ema_bull":          bool(cur_ema12 > cur_ema69),
        "htf_bias":          htf_bias,
        "exit_long":         exit_long,
        "exit_short":        exit_short,
        "bars_used":         n,
        "mean_lb_used":      mean_lb,
    })


@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    """Return the trained model's feature importance ranking."""
    model = load_model()
    if model is None:
        return jsonify({"error": "No model trained yet"}), 404

    importance = model.feature_importances_
    ranking = sorted(
        zip(FEATURE_COLUMNS, importance),
        key=lambda x: x[1],
        reverse=True
    )

    return jsonify({
        "features": [
            {"name": name, "importance": round(float(imp), 4)}
            for name, imp in ranking
        ]
    })


@app.route("/model-stats", methods=["GET"])
def model_stats():
    """Return current model statistics and training history."""
    model = load_model()

    # Get latest training insight from Supabase
    insights = supabase_query("insights", {
        "select": "description,created_at",
        "insight_type": "eq.ron_ml_training",
        "order": "created_at.desc",
        "limit": "1"
    })

    latest_training = None
    if insights:
        try:
            latest_training = json.loads(insights[0].get("description", "{}"))
            latest_training["trained_at"] = insights[0].get("created_at")
        except json.JSONDecodeError:
            pass

    return jsonify({
        "model_loaded": model is not None,
        "latest_training": latest_training,
        "feature_count": len(FEATURE_COLUMNS)
    })


@app.route("/analyse-setup", methods=["POST"])
def analyse_setup():
    """
    Full analysis of a trading setup — combines ML prediction
    with historical pattern stats.

    This is what powers the "Ask RON" intelligent responses
    about specific trade setups.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Get ML prediction
    prediction = predict_win_probability(data)

    # Get historical stats for this pattern + instrument combo
    symbol = data.get("symbol", "")
    pattern = data.get("pattern_active", "")
    session = data.get("session", "")

    historical = supabase_query("ron_platform_intelligence", {
        "select": "*",
        "symbol": f"eq.{symbol}",
        "limit": "10"
    })

    # Find matching pattern stats
    pattern_stats = None
    session_stats = None
    for row in historical:
        if row.get("pattern") == pattern and pattern:
            pattern_stats = row
        if row.get("session") == session and session:
            session_stats = row

    analysis = {
        "ml_prediction": prediction,
        "pattern_historical": {
            "win_rate": pattern_stats.get("win_rate") if pattern_stats else None,
            "total_trades": pattern_stats.get("total_signals") if pattern_stats else 0,
            "avg_pips_won": pattern_stats.get("avg_pips_won") if pattern_stats else None,
            "sample_users": pattern_stats.get("sample_size_users") if pattern_stats else 0
        } if pattern_stats else {"message": "No historical data for this pattern yet"},
        "session_performance": {
            "win_rate": session_stats.get("win_rate") if session_stats else None,
            "best_hour": session_stats.get("best_hour_utc") if session_stats else None
        } if session_stats else {"message": "No session data yet"},
        "overall_recommendation": "STRONG BUY" if prediction["probability"] >= 0.70
            else "BUY" if prediction["probability"] >= 0.60
            else "WAIT" if prediction["probability"] >= 0.45
            else "AVOID",
        "reasoning": generate_reasoning(data, prediction, pattern_stats, session_stats)
    }

    return jsonify(analysis)


def generate_reasoning(data, prediction, pattern_stats, session_stats) -> str:
    """Generate human-readable reasoning for RON's analysis."""
    parts = []

    prob = prediction["probability"]
    symbol = data.get("symbol", "Unknown")
    direction = data.get("direction", "Unknown")

    parts.append(f"RON ML Model rates this {symbol} {direction} setup at {prob*100:.0f}% probability of success.")

    adx = data.get("adx_at_entry", 0)
    if adx > 25:
        parts.append(f"ADX at {adx:.1f} confirms strong trend — good for directional trades.")
    elif adx < 18:
        parts.append(f"ADX at {adx:.1f} indicates weak trend — higher risk of false signal.")

    rsi = data.get("rsi_at_entry", 50)
    if direction == "BUY" and rsi > 70:
        parts.append(f"Caution: RSI at {rsi:.1f} is overbought territory.")
    elif direction == "SELL" and rsi < 30:
        parts.append(f"Caution: RSI at {rsi:.1f} is oversold territory.")

    pattern = data.get("pattern_active", "")
    if pattern_stats and pattern:
        wr = pattern_stats.get("win_rate", 0)
        total = pattern_stats.get("total_signals", 0)
        parts.append(f"{pattern} on {symbol}: platform win rate {wr*100:.0f}% across {total} trades.")

    session = data.get("session", "")
    if session_stats:
        swr = session_stats.get("win_rate", 0)
        parts.append(f"During {session} session: platform win rate {swr*100:.0f}%.")

    if prob >= 0.70:
        parts.append("This is a HIGH CONVICTION setup. Consider full position size.")
    elif prob >= 0.60:
        parts.append("Moderate probability. Consider reduced position size.")
    elif prob >= 0.45:
        parts.append("Below threshold. Consider waiting for better conditions.")
    else:
        parts.append("RON recommends AVOIDING this trade. Conditions unfavourable.")

    return " ".join(parts)


# ─── Yahoo Finance Data Ingestion ────────────────────────────
# Symbol mapping: GAINEDGE symbol → Yahoo Finance ticker
YAHOO_SYMBOLS = {
    "XAUUSD": "GC=F",          # Gold futures
    "US30": "YM=F",             # Dow Jones futures
    "NAS100": "NQ=F",           # Nasdaq futures
    "NZDUSD": "NZDUSD=X",      # NZD/USD forex
    "AUDUSD": "AUDUSD=X",      # AUD/USD forex
    "EURUSD": "EURUSD=X",      # EUR/USD forex
    "GBPUSD": "GBPUSD=X",      # GBP/USD forex
    "USDJPY": "USDJPY=X",      # USD/JPY forex
    "USDCHF": "USDCHF=X",      # USD/CHF forex
    "USDCAD": "USDCAD=X",      # USD/CAD forex
    "GBPJPY": "GBPJPY=X",      # GBP/JPY forex
    "EURJPY": "EURJPY=X",      # EUR/JPY forex
    "AUDJPY": "AUDJPY=X",      # AUD/JPY forex
    "EURGBP": "EURGBP=X",      # EUR/GBP forex
    "XAGUSD": "SI=F",          # Silver futures
    "USOIL": "CL=F",           # Crude Oil WTI
    "UKOIL": "BZ=F",           # Brent Oil
    "HK50": "^HSI",            # Hang Seng Index
    "GER40": "^GDAXI",         # DAX 40
    "UK100": "^FTSE",          # FTSE 100
    "JP225": "^N225",          # Nikkei 225
    "US500": "^GSPC",          # S&P 500
    "AUS200": "^AXJO",         # ASX 200
}


def fetch_yahoo_history(gainedge_symbol: str, period: str = "1y", interval: str = "1d") -> list:
    """
    Fetch historical OHLCV data from Yahoo Finance for a given instrument.
    period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    interval: 1m, 5m, 15m, 1h, 1d, 1wk, 1mo
    """
    import yfinance as yf

    yahoo_ticker = YAHOO_SYMBOLS.get(gainedge_symbol)
    if not yahoo_ticker:
        logger.warning(f"No Yahoo Finance mapping for {gainedge_symbol}")
        return []

    try:
        ticker = yf.Ticker(yahoo_ticker)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {yahoo_ticker}")
            return []

        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "symbol": gainedge_symbol,
                "timeframe": interval,
                "timestamp": idx.isoformat(),
                "open": round(float(row["Open"]), 5),
                "high": round(float(row["High"]), 5),
                "low": round(float(row["Low"]), 5),
                "close": round(float(row["Close"]), 5),
                "volume": int(row.get("Volume", 0))
            })

        logger.info(f"Fetched {len(candles)} candles for {gainedge_symbol} ({yahoo_ticker})")
        return candles

    except Exception as e:
        logger.error(f"Yahoo Finance error for {gainedge_symbol}: {e}")
        return []


def store_candles_in_supabase(candles: list) -> int:
    """Bulk insert candles via SECURITY DEFINER RPC (bypasses RLS with anon key).
    Falls back to direct insert if RPC not available."""
    if not candles:
        return 0

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    stored = 0

    for i in range(0, len(candles), 500):
        batch = candles[i:i + 500]
        # Use SECURITY DEFINER RPC — works with anon key, bypasses RLS
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/bulk_insert_candles",
            headers=headers,
            json={"candles": batch},
            timeout=30,
        )
        if resp.status_code == 200:
            try:
                stored += int(resp.json())
            except Exception:
                stored += len(batch)  # assume all stored if count unparseable
        else:
            # Fallback: direct insert (works if service_role key is set)
            logger.warning(f"RPC insert failed ({resp.status_code}), trying direct insert")
            direct_resp = requests.post(
                f"{SUPABASE_URL}/rest/v1/candle_history",
                headers={**headers, "Prefer": "resolution=ignore-duplicates,return=minimal"},
                json=batch,
                timeout=30,
            )
            if direct_resp.status_code in (200, 201, 204):
                stored += len(batch)
            else:
                logger.error(f"Direct insert also failed: {direct_resp.status_code} {direct_resp.text[:200]}")

    return stored


@app.route("/ingest/symbol", methods=["POST"])
def ingest_symbol():
    """
    Fetch historical data for a single instrument from Yahoo Finance
    and store in candle_history.

    JSON body: { "symbol": "XAUUSD", "period": "1y", "interval": "1d" }
    """
    api_key = request.headers.get("X-API-Key", "")
    if api_key != os.environ.get("RON_API_KEY", "gainedge-ron-2026"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    symbol = data.get("symbol", "")
    period = data.get("period", "1y")
    interval = data.get("interval", "1d")

    candles = fetch_yahoo_history(symbol, period, interval)
    stored = store_candles_in_supabase(candles)

    return jsonify({
        "symbol": symbol,
        "yahoo_ticker": YAHOO_SYMBOLS.get(symbol, "unknown"),
        "candles_fetched": len(candles),
        "candles_stored": stored,
        "period": period,
        "interval": interval
    })


@app.route("/ingest/all", methods=["POST"])
def ingest_all():
    """
    Fetch historical data for ALL mapped instruments from Yahoo Finance.
    This is the big one — populates RON's brain with years of data.

    JSON body: { "period": "2y", "interval": "1d" }
    """
    api_key = request.headers.get("X-API-Key", "")
    if api_key != os.environ.get("RON_API_KEY", "gainedge-ron-2026"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    period = data.get("period", "1y")
    interval = data.get("interval", "1d")

    results = {}
    total_candles = 0

    for symbol in YAHOO_SYMBOLS.keys():
        candles = fetch_yahoo_history(symbol, period, interval)
        stored = store_candles_in_supabase(candles)
        results[symbol] = {
            "fetched": len(candles),
            "stored": stored
        }
        total_candles += stored
        logger.info(f"Ingested {symbol}: {stored} candles")

    # Log the ingestion event
    supabase_insert("insights", {
        "insight_type": "ron_data_ingestion",
        "title": "Yahoo Finance Bulk Ingestion",
        "description": json.dumps({
            "total_candles": total_candles,
            "instruments": len(results),
            "period": period,
            "interval": interval,
            "results": results
        }),
        "created_at": datetime.utcnow().isoformat()
    })

    return jsonify({
        "status": "complete",
        "total_candles_stored": total_candles,
        "instruments_processed": len(results),
        "period": period,
        "interval": interval,
        "details": results
    })


@app.route("/ingest/available", methods=["GET"])
def ingest_available():
    """List all instruments available for Yahoo Finance ingestion."""
    return jsonify({
        "instruments": [
            {"gainedge_symbol": k, "yahoo_ticker": v}
            for k, v in YAHOO_SYMBOLS.items()
        ],
        "total": len(YAHOO_SYMBOLS),
        "periods": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        "intervals": ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"]
    })


@app.route("/ingest/economic-calendar", methods=["POST"])
def ingest_economic_calendar():
    """
    Fetch upcoming economic events from Finnhub and store them.
    RON uses this to know when high-impact events are coming.
    """
    api_key = request.headers.get("X-API-Key", "")
    if api_key != os.environ.get("RON_API_KEY", "gainedge-ron-2026"):
        return jsonify({"error": "Unauthorized"}), 401

    finnhub_key = os.environ.get("FINNHUB_API_KEY", "d0mlsg1r01qqqs5aa4h0d0mlsg1r01qqqs5aa4hg")

    today = datetime.utcnow().strftime("%Y-%m-%d")
    next_week = (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            f"https://finnhub.io/api/v1/calendar/economic",
            params={"from": today, "to": next_week, "token": finnhub_key}
        )
        if resp.status_code == 200:
            events = resp.json().get("economicCalendar", [])

            # Filter high-impact events
            high_impact = [e for e in events if e.get("impact", 0) >= 2]

            return jsonify({
                "total_events": len(events),
                "high_impact_events": len(high_impact),
                "events": high_impact[:20]  # Top 20 high impact
            })
    except Exception as e:
        logger.error(f"Finnhub calendar error: {e}")

    return jsonify({"error": "Failed to fetch calendar"}), 500


# ─── Alpha Vantage Economic Intelligence ─────────────────────
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "PDM6N3KVGIL3S8ZT")

@app.route("/intelligence/economic", methods=["GET"])
def economic_indicators():
    """Fetch key economic indicators — Fed rate, CPI, unemployment, treasury, GDP."""
    indicators = {}
    endpoints = [
        ("fed_funds_rate", "FEDERAL_FUNDS_RATE", {"interval": "monthly"}),
        ("cpi", "CPI", {"interval": "monthly"}),
        ("unemployment", "UNEMPLOYMENT", {}),
        ("treasury_10y", "TREASURY_YIELD", {"interval": "daily", "maturity": "10year"}),
        ("real_gdp", "REAL_GDP", {"interval": "quarterly"}),
    ]
    for name, func, extra_params in endpoints:
        try:
            params = {"function": func, "apikey": ALPHA_VANTAGE_KEY}
            params.update(extra_params)
            resp = requests.get("https://www.alphavantage.co/query", params=params)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    indicators[name] = {"value": data[0].get("value"), "date": data[0].get("date"),
                        "previous": data[1].get("value") if len(data) > 1 else None}
        except Exception as e:
            logger.error(f"{name} error: {e}")

    # Generate RON insight
    parts = []
    fed = indicators.get("fed_funds_rate", {})
    if fed and fed.get("value") and fed.get("previous"):
        direction = "rising" if float(fed["value"]) > float(fed["previous"]) else "falling"
        parts.append(f"Fed rate {fed['value']}% ({direction}). {'Rising rates pressure gold, strengthen USD.' if direction == 'rising' else 'Falling rates support gold, weaken USD.'}")
    cpi = indicators.get("cpi", {})
    if cpi and cpi.get("value") and cpi.get("previous"):
        parts.append(f"CPI {'rising — bullish gold, bearish equities' if float(cpi['value']) > float(cpi['previous']) else 'cooling — reduces rate hike urgency'}.")
    unemp = indicators.get("unemployment", {})
    if unemp and unemp.get("value"):
        parts.append(f"Unemployment {unemp['value']}%.")

    supabase_insert("insights", {"insight_type": "economic_indicators", "title": "Economic Update",
        "description": json.dumps(indicators), "created_at": datetime.utcnow().isoformat()})

    return jsonify({"indicators": indicators, "ron_insight": " ".join(parts), "fetched_at": datetime.utcnow().isoformat()})


@app.route("/intelligence/sentiment", methods=["GET"])
def market_sentiment():
    """Fetch AI-powered news sentiment scores per instrument from Alpha Vantage."""
    symbol = request.args.get("symbol", "")
    av_map = {"XAUUSD": "FOREX:XAU", "US30": "DJI", "NAS100": "NDX", "AUDUSD": "FOREX:AUD",
        "NZDUSD": "FOREX:NZD", "EURUSD": "FOREX:EUR", "GBPUSD": "FOREX:GBP", "USOIL": "CRUDE_OIL"}
    tickers = av_map.get(symbol, symbol) if symbol else ",".join(av_map.values())

    try:
        resp = requests.get("https://www.alphavantage.co/query", params={
            "function": "NEWS_SENTIMENT", "tickers": tickers, "limit": "20", "apikey": ALPHA_VANTAGE_KEY})
        if resp.status_code == 200:
            feed = resp.json().get("feed", [])
            summary = {}
            for article in feed:
                for td in article.get("ticker_sentiment", []):
                    t = td.get("ticker", "")
                    score = float(td.get("ticker_sentiment_score", 0))
                    if t not in summary:
                        summary[t] = {"scores": [], "count": 0}
                    summary[t]["scores"].append(score)
                    summary[t]["count"] += 1
            for t, d in summary.items():
                avg = sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0
                d["avg_sentiment"] = round(avg, 4)
                d["overall"] = "Bullish" if avg > 0.15 else ("Bearish" if avg < -0.15 else "Neutral")
                d["strength"] = "Strong" if abs(avg) > 0.35 else ("Moderate" if abs(avg) > 0.15 else "Weak")
                d["articles"] = d.pop("count")
                del d["scores"]
            return jsonify({"sentiment": summary, "articles_analysed": len(feed), "fetched_at": datetime.utcnow().isoformat()})
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
    return jsonify({"error": "Failed to fetch sentiment"}), 500


@app.route("/intelligence/fear-greed", methods=["GET"])
def fear_greed_index():
    """Fetch CNN Fear & Greed Index. Extreme fear = buy, extreme greed = caution."""
    try:
        resp = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            data = resp.json()
            fg = data.get("fear_and_greed", {})
            score = fg.get("score", 50)
            rating = fg.get("rating", "Neutral")
            if score <= 25: signal = "EXTREME FEAR — strong buy signal historically. Markets oversold."
            elif score <= 40: signal = "FEAR — look for buy opportunities on dips."
            elif score <= 60: signal = "NEUTRAL — trade the technicals."
            elif score <= 75: signal = "GREED — be selective, tighten stops."
            else: signal = "EXTREME GREED — warning signal. Consider reducing exposure."
            result = {"score": score, "rating": rating, "previous_close": fg.get("previous_close", 50),
                "one_week_ago": fg.get("previous_1_week", 50), "one_month_ago": fg.get("previous_1_month", 50),
                "ron_signal": signal, "fetched_at": datetime.utcnow().isoformat()}
            supabase_insert("insights", {"insight_type": "fear_greed_index",
                "title": f"Fear & Greed: {score} ({rating})", "description": json.dumps(result),
                "created_at": datetime.utcnow().isoformat()})
            return jsonify(result)
    except Exception as e:
        logger.error(f"Fear & Greed error: {e}")
    return jsonify({"error": "Failed to fetch Fear & Greed Index"}), 500


@app.route("/intelligence/full-briefing", methods=["GET"])
def full_briefing():
    """RON's complete intelligence briefing — all data sources combined."""
    port = os.environ.get("PORT", 5000)
    briefing = {"timestamp": datetime.utcnow().isoformat(), "sections": {}}
    for name, path in [("economic", "/intelligence/economic"), ("fear_greed", "/intelligence/fear-greed"), ("sentiment", "/intelligence/sentiment")]:
        try:
            r = requests.get(f"http://localhost:{port}{path}")
            briefing["sections"][name] = r.json() if r.status_code == 200 else {"status": "unavailable"}
        except:
            briefing["sections"][name] = {"status": "unavailable"}
    briefing["ml_model_loaded"] = load_model() is not None
    return jsonify(briefing)


# ─── GitHub CSV Historical Data Ingestion ─────────────────────
GITHUB_CSV_REPO = "https://raw.githubusercontent.com/carlfalc/ron-ml/main"

# Map Dukascopy file prefixes to GAINEDGE symbols
DUKASCOPY_SYMBOL_MAP = {
    # Metals
    "XAU-USD": "XAUUSD",
    "XAU-AUD": "XAUAUD",
    "XAG-USD": "XAGUSD",
    # Majors
    "EUR-USD": "EURUSD",
    "GBP-USD": "GBPUSD",
    "USD-JPY": "USDJPY",
    "USD-CHF": "USDCHF",
    "USD-CAD": "USDCAD",
    "AUD-USD": "AUDUSD",
    "NZD-USD": "NZDUSD",
    # Crosses
    "EUR-GBP": "EURGBP",
    "EUR-JPY": "EURJPY",
    "EUR-AUD": "EURAUD",
    "EUR-CAD": "EURCAD",
    "EUR-CHF": "EURCHF",
    "EUR-NZD": "EURNZD",
    "GBP-JPY": "GBPJPY",
    "GBP-AUD": "GBPAUD",
    "GBP-CAD": "GBPCAD",
    "GBP-CHF": "GBPCHF",
    "GBP-NZD": "GBPNZD",
    "AUD-JPY": "AUDJPY",
    "AUD-CAD": "AUDCAD",
    "AUD-CHF": "AUDCHF",
    "AUD-NZD": "AUDNZD",
    "NZD-JPY": "NZDJPY",
    "NZD-CAD": "NZDCAD",
    "NZD-CHF": "NZDCHF",
    "CAD-JPY": "CADJPY",
    "CAD-CHF": "CADCHF",
    "CHF-JPY": "CHFJPY",
    # Commodities
    "LIGHT.CMD-USD": "USOIL",
    "LIGHT-USD": "USOIL",
    "BRENT.CMD-USD": "UKOIL",
    "GAS.CMD-USD": "XNGUSD",
    "COPPER.CMD-USD": "XCUUSD",
    # Indices
    "USA500.IDX-USD": "US500",
    "USA500IDX-USD": "US500",
    "USA30.IDX-USD": "US30",
    "USATECH.IDX-USD": "NAS100",
    "JPN.IDX-JPY": "JP225",
    "GER.IDX-EUR": "GER40",
    "GBR.IDX-GBP": "UK100",
    "HKG.IDX-HKD": "HK50",
    "AUS.IDX-AUD": "AUS200",
}


def parse_dukascopy_csv(csv_text: str, symbol: str) -> list:
    """Parse Dukascopy CSV format into candle records."""
    candles = []
    lines = csv_text.strip().split("\n")

    for line in lines:
        try:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            # Skip header rows
            if parts[0].startswith("time") or parts[0].startswith("Time") or parts[0].startswith("date"):
                continue

            timestamp = parts[0].strip()
            open_p = float(parts[1].strip())
            high_p = float(parts[2].strip())
            low_p = float(parts[3].strip())
            close_p = float(parts[4].strip())
            volume = int(float(parts[5].strip())) if parts[5].strip() else 0

            # Skip zero-price candles
            if open_p == 0 or close_p == 0:
                continue

            candles.append({
                "symbol": symbol,
                "timeframe": "1m",
                "timestamp": timestamp,
                "open": round(open_p, 5),
                "high": round(high_p, 5),
                "low": round(low_p, 5),
                "close": round(close_p, 5),
                "volume": volume
            })
        except (ValueError, IndexError) as e:
            continue

    return candles


@app.route("/ingest/github-csv", methods=["POST"])
def ingest_github_csv():
    """
    Fetch a specific CSV file from the ron-ml GitHub repo and import into candle_history.
    JSON body: { "filename": "XAU-USD_Minute_2016-01-01_UTC.csv" }
    """
    api_key = request.headers.get("X-API-Key", "")
    if api_key != os.environ.get("RON_API_KEY", "gainedge-ron-2026"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    filename = data.get("filename", "")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    # Determine GAINEDGE symbol from filename
    symbol = None
    for prefix, sym in DUKASCOPY_SYMBOL_MAP.items():
        if filename.startswith(prefix):
            symbol = sym
            break

    if not symbol:
        return jsonify({"error": f"Cannot determine symbol from filename: {filename}"}), 400

    # Fetch CSV from GitHub
    url = f"{GITHUB_CSV_REPO}/{filename}"
    logger.info(f"Fetching {url}")

    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code != 200:
            return jsonify({"error": f"Failed to fetch {filename}: {resp.status_code}"}), 400

        candles = parse_dukascopy_csv(resp.text, symbol)
        logger.info(f"Parsed {len(candles)} candles from {filename}")

        stored = store_candles_in_supabase(candles)

        return jsonify({
            "filename": filename,
            "symbol": symbol,
            "candles_parsed": len(candles),
            "candles_stored": stored
        })

    except Exception as e:
        logger.error(f"GitHub CSV ingestion error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ingest/github-all", methods=["POST"])
def ingest_all_github_csvs():
    """
    Fetch ALL CSV files from the ron-ml GitHub repo and import them.
    This loads all historical Dukascopy data into RON's brain.
    """
    api_key = request.headers.get("X-API-Key", "")
    if api_key != os.environ.get("RON_API_KEY", "gainedge-ron-2026"):
        return jsonify({"error": "Unauthorized"}), 401

    # List known CSV files from the repo
    csv_files = []
    try:
        resp = requests.get("https://api.github.com/repos/carlfalc/ron-ml/contents/",
            headers={"Accept": "application/vnd.github.v3+json"})
        if resp.status_code == 200:
            for item in resp.json():
                if item.get("name", "").endswith(".csv"):
                    csv_files.append(item["name"])
    except Exception as e:
        logger.error(f"GitHub API error: {e}")
        return jsonify({"error": "Failed to list repo files"}), 500

    if not csv_files:
        return jsonify({"error": "No CSV files found in repo"}), 404

    results = {}
    total_stored = 0

    for filename in csv_files:
        # Determine symbol
        symbol = None
        for prefix, sym in DUKASCOPY_SYMBOL_MAP.items():
            if filename.startswith(prefix):
                symbol = sym
                break
        if not symbol:
            results[filename] = {"status": "skipped", "reason": "unknown symbol"}
            continue

        try:
            url = f"{GITHUB_CSV_REPO}/{filename}"
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200:
                candles = parse_dukascopy_csv(resp.text, symbol)
                stored = store_candles_in_supabase(candles)
                results[filename] = {"symbol": symbol, "parsed": len(candles), "stored": stored}
                total_stored += stored
                logger.info(f"Ingested {filename}: {stored} candles for {symbol}")
            else:
                results[filename] = {"status": "failed", "http_code": resp.status_code}
        except Exception as e:
            results[filename] = {"status": "error", "message": str(e)}

    return jsonify({
        "status": "complete",
        "total_files": len(csv_files),
        "total_candles_stored": total_stored,
        "details": results
    })


# ═══════════════════════════════════════════════════════════════════════════════
# /backtest — Historical RON v3 backtest with IS/OOS split
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_candles_range(symbol: str, timeframe: str, start_iso: str, end_iso: str,
                          page_size: int = 1000) -> list:
    """Paginated fetch of candle_history rows in a date range. Uses list-of-tuples
    params so we can filter `timestamp` twice (gte + lt)."""
    url = f"{SUPABASE_URL}/rest/v1/candle_history"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    all_rows = []
    offset = 0
    while True:
        params = [
            ("select",    "timestamp,open,high,low,close,volume"),
            ("symbol",    f"eq.{symbol}"),
            ("timeframe", f"eq.{timeframe}"),
            ("timestamp", f"gte.{start_iso}"),
            ("timestamp", f"lt.{end_iso}"),
            ("order",     "timestamp.asc"),
            ("limit",     str(page_size)),
            ("offset",    str(offset)),
        ]
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        if resp.status_code != 200:
            logger.error(f"Supabase fetch failed: {resp.status_code} {resp.text[:200]}")
            break
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


def _aggregate_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1-minute bars to a higher timeframe (e.g. '15min', '1h')."""
    df = df_1m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    agg = df.resample(rule, label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"]).reset_index()
    return agg


def _compute_indicators_full(df: pd.DataFrame, mean_lb: int = 360,
                             ema_fast: int = 12, ema_slow: int = 69,
                             candle_type: str = "HA"):
    """Compute DLO, Squeeze, HA/standard, EMA(fast)/EMA(slow) once on the full series.
    No look-ahead — each value at index i depends only on data up to i.
    Defaults match V3 production (12/69 EMA, Heikin Ashi)."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    open_ = df["open"]

    dlo_s, _              = _calc_dlo(high, low, close, mean_lb=mean_lb)
    sqz_on, sqz_off, sqz_val = _calc_squeeze(close, high, low)

    if str(candle_type).upper() == "HA":
        ha_close_vals = ((open_ + high + low + close) / 4.0).values
        ha_open_vals  = np.zeros(len(df))
        ha_open_vals[0] = (open_.iloc[0] + close.iloc[0]) / 2.0
        for i in range(1, len(df)):
            ha_open_vals[i] = (ha_open_vals[i - 1] + ha_close_vals[i - 1]) / 2.0
        candle_bull = pd.Series(ha_close_vals > ha_open_vals, index=df.index)
    else:
        candle_bull = pd.Series(close.values > open_.values, index=df.index)

    ema_fast_s = _ema_s(close, int(ema_fast))
    ema_slow_s = _ema_s(close, int(ema_slow))

    # ATR(14) — Wilder
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = _rma(tr.fillna(0), 14)

    return {
        "dlo":      dlo_s,
        "sqz_on":   sqz_on,
        "sqz_off":  sqz_off,
        "sqz_val":  sqz_val,
        "ha_bull":  candle_bull,   # name kept for compat; HA or standard per candle_type
        "ema_fast": ema_fast_s,
        "ema_slow": ema_slow_s,
        "atr14":    atr14,
    }


def _signal_at_index(ind: dict, t: int, fire_lookback: int,
                     conviction_a: float, conviction_b: float,
                     min_tier: str, require_sqz_fire: bool,
                     ema_filter: bool = False) -> dict:
    """Compute the v3 signal using only indicator values up to index t (no peek)."""
    if t < 2:
        return {"signal": "HOLD", "ron_action": "HOLD", "tier": "NONE"}

    cur_dlo, prev_dlo, prev2_dlo = float(ind["dlo"].iloc[t]), float(ind["dlo"].iloc[t-1]), float(ind["dlo"].iloc[t-2])
    cur_sqz_val, prev_sqz_val    = float(ind["sqz_val"].iloc[t]), float(ind["sqz_val"].iloc[t-1])
    cur_sqz_on  = bool(ind["sqz_on"].iloc[t])
    cur_sqz_off = bool(ind["sqz_off"].iloc[t])
    cur_ha_bull = bool(ind["ha_bull"].iloc[t])
    cur_ema_fast = float(ind["ema_fast"].iloc[t])
    cur_ema_slow = float(ind["ema_slow"].iloc[t])
    ema_bull = cur_ema_fast > cur_ema_slow

    if any(np.isnan(v) for v in [cur_dlo, cur_sqz_val]):
        return {"signal": "HOLD", "ron_action": "HOLD", "tier": "NONE"}

    sqz_just_fired = False
    if cur_sqz_off and t > fire_lookback:
        sqz_just_fired = bool(ind["sqz_on"].iloc[t-fire_lookback:t].any())
    squeeze_state = "ON" if cur_sqz_on else ("FIRED" if sqz_just_fired else "OFF")

    dlo_rising  = cur_dlo > prev_dlo > prev2_dlo
    dlo_falling = cur_dlo < prev_dlo < prev2_dlo
    sqz_accel_up = cur_sqz_val > 0 and cur_sqz_val > prev_sqz_val
    sqz_accel_dn = cur_sqz_val < 0 and cur_sqz_val < prev_sqz_val

    bull_aligned = cur_dlo > 0 and cur_sqz_val > 0
    bear_aligned = cur_dlo < 0 and cur_sqz_val < 0
    sqz_fire_ok  = sqz_just_fired if require_sqz_fire else True

    a_bull = bull_aligned and cur_dlo >  conviction_a and sqz_accel_up and sqz_fire_ok
    a_bear = bear_aligned and cur_dlo < -conviction_a and sqz_accel_dn and sqz_fire_ok
    b_bull = bull_aligned and sqz_accel_up and dlo_rising  and cur_dlo >  conviction_b and not a_bull
    b_bear = bear_aligned and sqz_accel_dn and dlo_falling and cur_dlo < -conviction_b and not a_bear

    direction, tier = None, "NONE"
    tier_rank = {"A": 0, "B": 1}.get(min_tier, 1)

    if a_bull:                          direction, tier = "BUY",  "A"
    elif a_bear:                        direction, tier = "SELL", "A"
    elif b_bull and tier_rank >= 1:     direction, tier = "BUY",  "B"
    elif b_bear and tier_rank >= 1:     direction, tier = "SELL", "B"

    ha_confirms = (direction == "BUY" and cur_ha_bull) or (direction == "SELL" and not cur_ha_bull)
    ema_confirms = True
    if ema_filter and direction is not None:
        ema_confirms = (direction == "BUY" and ema_bull) or (direction == "SELL" and not ema_bull)
    ron_action  = "EXECUTE" if (direction is not None and ha_confirms and ema_confirms) else "HOLD"

    return {
        "signal":        direction or "HOLD",
        "ron_action":    ron_action,
        "tier":          tier,
        "dlo":           round(cur_dlo, 4),
        "squeeze_state": squeeze_state,
        "ha_bull":       cur_ha_bull,
    }


def _simulate_trade(bars: pd.DataFrame, t_signal: int, direction: str,
                    entry: float, sl: float, tp: float, max_hold: int) -> dict:
    """Forward-scan bars looking for SL or TP hit. Conservative: SL wins ties."""
    end_idx = min(t_signal + 1 + max_hold, len(bars) - 1)
    for i in range(t_signal + 1, end_idx + 1):
        bar = bars.iloc[i]
        bar_high = float(bar["high"])
        bar_low  = float(bar["low"])
        if direction == "BUY":
            sl_hit = bar_low  <= sl
            tp_hit = bar_high >= tp
        else:
            sl_hit = bar_high >= sl
            tp_hit = bar_low  <= tp
        if sl_hit and tp_hit:
            return {"exit": sl, "exit_idx": i, "exit_reason": "sl"}
        if sl_hit:
            return {"exit": sl, "exit_idx": i, "exit_reason": "sl"}
        if tp_hit:
            return {"exit": tp, "exit_idx": i, "exit_reason": "tp"}
    timeout_idx = min(t_signal + max_hold, len(bars) - 1)
    return {"exit": float(bars.iloc[timeout_idx]["close"]),
            "exit_idx": timeout_idx, "exit_reason": "timeout"}


def _xau_pip() -> float:
    return 0.1


def _xau_pip_usd_per_lot() -> float:
    return 10.0  # $0.10 move × 100 oz/lot = $10/pip per standard lot


def _compute_metrics(trades: list, equity_curve: list, starting_balance: float) -> dict:
    """Build the metrics block for a slice of trades."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "avg_win_pips": 0, "avg_loss_pips": 0,
            "profit_factor": 0, "max_consecutive_losses": 0, "max_drawdown_pct": 0,
            "sharpe_ratio": 0, "final_equity": starting_balance,
            "by_tier": {"A": {}, "B": {}},
            "trade_duration_dist": [], "dlo_dist_at_entry": [],
            "best_trade": None, "worst_trade": None,
        }

    pnl_usd = [t["usd"] for t in trades]
    wins    = [t for t in trades if t["usd"] > 0]
    losses  = [t for t in trades if t["usd"] <= 0]

    win_rate = round(100 * len(wins) / len(trades), 2)
    avg_win_pips  = round(np.mean([t["pips"] for t in wins]), 2)  if wins   else 0
    avg_loss_pips = round(np.mean([t["pips"] for t in losses]), 2) if losses else 0
    gross_win  = sum(t["usd"] for t in wins)
    gross_loss = abs(sum(t["usd"] for t in losses))
    profit_factor = round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf") if gross_win > 0 else 0

    max_consec_loss = 0
    cur_streak = 0
    for t in trades:
        if t["usd"] <= 0:
            cur_streak += 1
            max_consec_loss = max(max_consec_loss, cur_streak)
        else:
            cur_streak = 0

    if equity_curve:
        eq_vals = np.array([e["equity"] for e in equity_curve])
        peak = np.maximum.accumulate(eq_vals)
        dd = (peak - eq_vals) / peak
        max_dd_pct = round(100 * float(dd.max()), 2)
        final_equity = round(float(eq_vals[-1]), 2)
        rets = np.diff(eq_vals) / eq_vals[:-1]
        sharpe = 0
        if len(rets) > 1 and rets.std() > 0:
            sharpe = round(float(rets.mean() / rets.std() * np.sqrt(252)), 3)
    else:
        max_dd_pct = 0
        final_equity = starting_balance
        sharpe = 0

    def tier_metrics(tier_letter: str) -> dict:
        ts = [t for t in trades if t["tier"] == tier_letter]
        if not ts: return {"total_trades": 0, "win_rate": 0, "profit_factor": 0, "avg_pips": 0}
        ws = [t for t in ts if t["usd"] > 0]
        ls = [t for t in ts if t["usd"] <= 0]
        gw = sum(t["usd"] for t in ws)
        gl = abs(sum(t["usd"] for t in ls))
        return {
            "total_trades": len(ts),
            "win_rate":     round(100 * len(ws) / len(ts), 2),
            "profit_factor": round(gw / gl, 3) if gl > 0 else float("inf") if gw > 0 else 0,
            "avg_pips":     round(np.mean([t["pips"] for t in ts]), 2),
        }

    duration_buckets = [(0, 4), (4, 12), (12, 24), (24, 96)]
    dur_dist = []
    for lo, hi in duration_buckets:
        count = sum(1 for t in trades if lo <= t["duration_bars"] / 4 < hi)
        dur_dist.append({"bucket": f"{lo}-{hi}h", "count": count})

    dlo_buckets = [(i / 10, (i + 1) / 10) for i in range(-5, 5)]
    dlo_dist = []
    for lo, hi in dlo_buckets:
        in_bucket = [t for t in trades if lo <= t["dlo"] < hi]
        dlo_dist.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "count":  len(in_bucket),
            "by_tier": {
                "A": sum(1 for t in in_bucket if t["tier"] == "A"),
                "B": sum(1 for t in in_bucket if t["tier"] == "B"),
            },
        })

    best  = max(trades, key=lambda t: t["usd"])
    worst = min(trades, key=lambda t: t["usd"])

    return {
        "total_trades":          len(trades),
        "win_rate":              win_rate,
        "avg_win_pips":          avg_win_pips,
        "avg_loss_pips":         avg_loss_pips,
        "profit_factor":         profit_factor,
        "max_consecutive_losses": max_consec_loss,
        "max_drawdown_pct":      max_dd_pct,
        "sharpe_ratio":          sharpe,
        "final_equity":          final_equity,
        "by_tier":               {"A": tier_metrics("A"), "B": tier_metrics("B")},
        "trade_duration_dist":   dur_dist,
        "dlo_dist_at_entry":     dlo_dist,
        "best_trade":            best,
        "worst_trade":           worst,
    }


def _verdict(is_metrics: dict, oos_metrics: dict, combined: dict) -> tuple:
    """Apply Lovable's go/no-go thresholds."""
    issues = []
    if is_metrics["total_trades"] < 30:
        issues.append(f"In-sample only {is_metrics['total_trades']} trades — sample too small")
    if oos_metrics["total_trades"] < 15:
        issues.append(f"Out-of-sample only {oos_metrics['total_trades']} trades — sample too small")
    if is_metrics.get("win_rate", 0) <= 55:
        issues.append(f"IS win rate {is_metrics.get('win_rate', 0)}% ≤ 55% threshold")
    if oos_metrics.get("win_rate", 0) <= 50:
        issues.append(f"OOS win rate {oos_metrics.get('win_rate', 0)}% ≤ 50% threshold — engine may overfit")
    if combined.get("profit_factor", 0) <= 1.5:
        issues.append(f"Profit factor {combined.get('profit_factor', 0)} ≤ 1.5 — try atr_tp_mult=3.0")
    if combined.get("max_drawdown_pct", 0) > 20:
        issues.append(f"Max drawdown {combined.get('max_drawdown_pct', 0)}% > 20% — reduce risk_per_trade_pct")
    a_count = combined.get("by_tier", {}).get("A", {}).get("total_trades", 0)
    if a_count < 10 and combined["total_trades"] >= 30:
        issues.append(f"Tier A only {a_count} trades — squeeze rule may be too restrictive")

    verdict = "production_ready" if not issues else "needs_tuning"
    return verdict, issues


@app.route("/backtest", methods=["POST"])
def backtest():
    """Historical RON v3 backtest. Pulls 1m candles from candle_history,
    aggregates to 15m + 1H, runs DLO+Squeeze+HA+EMA bar-by-bar, simulates
    trades with ATR SL/TP, returns full metrics + IS/OOS verdict."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != os.environ.get("RON_API_KEY", "gainedge-ron-2026"):
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json() or {}
    symbol         = body.get("symbol", "XAUUSD")
    timeframe      = body.get("timeframe", "15m")
    htf_timeframe  = body.get("htf_timeframe", "1h")
    start          = body.get("start")
    end            = body.get("end")
    is_split       = body.get("in_sample_split")
    warmup_bars    = int(body.get("warmup_bars", 400))

    cfg = body.get("config", {})
    starting_balance   = float(cfg.get("starting_balance", 10000))
    risk_pct           = float(cfg.get("risk_per_trade_pct", 1.0))
    atr_sl_mult        = float(cfg.get("atr_sl_mult", 1.5))
    atr_tp_mult        = float(cfg.get("atr_tp_mult", 2.5))
    min_tier           = str(cfg.get("min_tier", "B")).upper()
    spread_usd         = float(cfg.get("spread_usd", 0.30))
    max_hold_bars      = int(cfg.get("max_hold_bars", 96))
    fire_lookback      = int(cfg.get("fire_lookback", 10))
    conviction_a       = float(cfg.get("conviction_threshold", 0.25))
    conviction_b       = float(cfg.get("min_b_conviction", 0.15))
    require_sqz_fire   = bool(cfg.get("require_squeeze_fire", True))
    ema_fast           = int(cfg.get("ema_fast", 12))      # V3 default 12
    ema_slow           = int(cfg.get("ema_slow", 69))      # V3 default 69
    candle_type        = str(cfg.get("candle_type", "HA")).upper()  # V3 default HA
    ema_filter         = bool(cfg.get("ema_filter", False))         # V3 production: informational only

    if not start or not end:
        return jsonify({"error": "start and end required"}), 400
    if not is_split:
        is_split = start  # all OOS if no split specified

    logger.info(f"Backtest start: {symbol} {timeframe} from {start} to {end}")

    # 1. Pull 1m bars from candle_history (source of truth for granular data)
    rows_1m = _fetch_candles_range(symbol, "1m", start, end, page_size=1000)
    if len(rows_1m) < 1000:
        # Try pulling pre-aggregated bars if 1m sparse
        rows_native = _fetch_candles_range(symbol, timeframe, start, end, page_size=1000)
        if len(rows_native) < 200:
            return jsonify({
                "error": "insufficient candle data",
                "rows_1m": len(rows_1m),
                "rows_native": len(rows_native),
                "hint": "ingest 1m XAUUSD via /ingest/github-csv first"
            }), 400
        df_15m = pd.DataFrame(rows_native)
        df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
        df_15m = df_15m.sort_values("timestamp").reset_index(drop=True)
        df_1h  = None
    else:
        df_1m = pd.DataFrame(rows_1m)
        for col in ("open", "high", "low", "close", "volume"):
            df_1m[col] = pd.to_numeric(df_1m[col], errors="coerce")
        df_15m = _aggregate_ohlcv(df_1m, "15min")
        df_1h  = _aggregate_ohlcv(df_1m, "1h")

    n = len(df_15m)
    if n < warmup_bars + 100:
        return jsonify({
            "error": "insufficient bars after aggregation",
            "bars_15m": n,
            "warmup_required": warmup_bars,
        }), 400

    logger.info(f"Backtest: aggregated to {n} 15m bars + {len(df_1h) if df_1h is not None else 0} 1h bars")

    # 2. Compute indicators once on full series (with optional overrides)
    mean_lb = min(360, max(20, n - 30))
    ind = _compute_indicators_full(df_15m, mean_lb=mean_lb,
                                   ema_fast=ema_fast, ema_slow=ema_slow,
                                   candle_type=candle_type)

    # 3. Bar-by-bar loop
    trades = []
    equity = starting_balance
    equity_curve = [{"ts": str(df_15m.iloc[warmup_bars]["timestamp"]), "equity": equity}]

    open_position = None  # dict with {entry_idx, direction, sl, tp, entry, lots, tier, dlo, sqz}

    for t in range(warmup_bars, n - 1):
        bar = df_15m.iloc[t]
        ts  = bar["timestamp"]

        # Close open position if SL/TP hit on current bar
        if open_position is not None:
            d  = open_position
            bar_high = float(bar["high"])
            bar_low  = float(bar["low"])
            if d["direction"] == "BUY":
                sl_hit = bar_low  <= d["sl"]
                tp_hit = bar_high >= d["tp"]
            else:
                sl_hit = bar_high >= d["sl"]
                tp_hit = bar_low  <= d["tp"]
            exit_now = None; reason = None
            if sl_hit:                       exit_now, reason = d["sl"], "sl"
            elif tp_hit:                     exit_now, reason = d["tp"], "tp"
            elif (t - d["entry_idx"]) >= max_hold_bars:
                exit_now, reason = float(bar["close"]), "timeout"

            if exit_now is not None:
                pip_size = _xau_pip()
                pip_usd  = _xau_pip_usd_per_lot()
                if d["direction"] == "BUY":
                    pips = (exit_now - d["entry"]) / pip_size
                else:
                    pips = (d["entry"] - exit_now) / pip_size
                usd  = pips * pip_usd * d["lots"]
                equity += usd
                trades.append({
                    "ts":            str(d["entry_ts"]),
                    "exit_ts":       str(ts),
                    "tier":          d["tier"],
                    "direction":     d["direction"],
                    "dlo":           d["dlo"],
                    "squeeze":       d["sqz"],
                    "entry":         round(d["entry"], 5),
                    "exit":          round(exit_now, 5),
                    "pips":          round(pips, 2),
                    "usd":           round(usd, 2),
                    "duration_bars": t - d["entry_idx"],
                    "exit_reason":   reason,
                    "lots":          d["lots"],
                })
                equity_curve.append({"ts": str(ts), "equity": round(equity, 2)})
                open_position = None

        # Look for new entry
        if open_position is None:
            sig = _signal_at_index(ind, t, fire_lookback, conviction_a, conviction_b,
                                   min_tier, require_sqz_fire, ema_filter=ema_filter)
            if sig["ron_action"] == "EXECUTE":
                next_bar = df_15m.iloc[t + 1]
                raw_entry = float(next_bar["open"])
                direction = sig["signal"]
                # Spread: BUY pays the ask, SELL receives the bid
                entry = raw_entry + spread_usd if direction == "BUY" else raw_entry - spread_usd

                atr_val = float(ind["atr14"].iloc[t])
                if np.isnan(atr_val) or atr_val <= 0:
                    continue
                sl_dist = atr_val * atr_sl_mult
                tp_dist = atr_val * atr_tp_mult
                sl = entry - sl_dist if direction == "BUY" else entry + sl_dist
                tp = entry + tp_dist if direction == "BUY" else entry - tp_dist

                pip_size = _xau_pip()
                pip_usd  = _xau_pip_usd_per_lot()
                sl_pips  = sl_dist / pip_size
                risk_usd = equity * (risk_pct / 100)
                raw_lots = risk_usd / max(sl_pips * pip_usd, 1e-6)
                lots     = max(0.01, min(0.50, round(raw_lots * 100) / 100))

                open_position = {
                    "entry_idx": t + 1,
                    "entry_ts":  next_bar["timestamp"],
                    "direction": direction,
                    "entry":     entry,
                    "sl":        round(sl, 5),
                    "tp":        round(tp, 5),
                    "lots":      lots,
                    "tier":      sig["tier"],
                    "dlo":       sig["dlo"],
                    "sqz":       sig["squeeze_state"],
                }

    # 4. Split trades IS / OOS
    is_split_ts = pd.to_datetime(is_split)
    is_trades  = [t for t in trades if pd.to_datetime(t["ts"]) <  is_split_ts]
    oos_trades = [t for t in trades if pd.to_datetime(t["ts"]) >= is_split_ts]

    # Build per-slice equity curves for accurate per-slice DD/Sharpe
    is_eq  = [e for e in equity_curve if pd.to_datetime(e["ts"]) <  is_split_ts]
    oos_eq = [e for e in equity_curve if pd.to_datetime(e["ts"]) >= is_split_ts]
    if not is_eq:  is_eq  = [{"ts": start, "equity": starting_balance}]
    if not oos_eq:
        oos_eq = [{"ts": is_split, "equity": is_eq[-1]["equity"] if is_eq else starting_balance}]

    is_metrics  = _compute_metrics(is_trades,  is_eq,  starting_balance)
    oos_metrics = _compute_metrics(oos_trades, oos_eq, is_eq[-1]["equity"] if is_eq else starting_balance)
    combined    = _compute_metrics(trades,     equity_curve, starting_balance)

    verdict, issues = _verdict(is_metrics, oos_metrics, combined)

    return jsonify({
        "run_id": str(uuid.uuid4()),
        "config": {
            "symbol": symbol, "timeframe": timeframe, "htf_timeframe": htf_timeframe,
            "start": start, "end": end, "in_sample_split": is_split,
            "warmup_bars": warmup_bars,
            "starting_balance": starting_balance, "risk_per_trade_pct": risk_pct,
            "atr_sl_mult": atr_sl_mult, "atr_tp_mult": atr_tp_mult,
            "min_tier": min_tier, "spread_usd": spread_usd,
            "max_hold_bars": max_hold_bars,
            "ema_fast": ema_fast, "ema_slow": ema_slow,
            "candle_type": candle_type, "ema_filter": ema_filter,
            "conviction_threshold": conviction_a, "min_b_conviction": conviction_b,
            "require_squeeze_fire": require_sqz_fire,
        },
        "data_window": {
            "bars_used": n,
            "earliest":  str(df_15m.iloc[0]["timestamp"]),
            "latest":    str(df_15m.iloc[-1]["timestamp"]),
        },
        "in_sample":     is_metrics,
        "out_of_sample": oos_metrics,
        "combined":      combined,
        "trades":        trades,
        "equity_curve":  equity_curve,
        "verdict":       verdict,
        "issues":        issues,
    })


# ─── Dukascopy direct datafeed (no manual CSVs) ───────────────────────────
DUKASCOPY_BASE = "https://datafeed.dukascopy.com/datafeed"

# (dukascopy_symbol, point_factor)
# point_factor: 3 means raw price / 1000;  5 means raw price / 100000
DUKASCOPY_INSTRUMENT_MAP = {
    "XAUUSD": ("XAUUSD", 3),
    "XAUAUD": ("XAUAUD", 3),
    "XAGUSD": ("XAGUSD", 3),
    "EURUSD": ("EURUSD", 5),
    "GBPUSD": ("GBPUSD", 5),
    "USDJPY": ("USDJPY", 3),
    "AUDUSD": ("AUDUSD", 5),
    "NZDUSD": ("NZDUSD", 5),
    "USDCAD": ("USDCAD", 5),
    "EURJPY": ("EURJPY", 3),
    "GBPJPY": ("GBPJPY", 3),
    "AUDJPY": ("AUDJPY", 3),
    "EURGBP": ("EURGBP", 5),
    "EURNZD": ("EURNZD", 5),
    "GBPCAD": ("GBPCAD", 5),
    "AUDCAD": ("AUDCAD", 5),
    "AUDNZD": ("AUDNZD", 5),
    "NZDCAD": ("NZDCAD", 5),
}


def _fetch_dukascopy_hour(dk_sym: str, point_factor: int,
                          year: int, month0: int, day: int, hour: int,
                          timeout: int = 12) -> Optional[list]:
    """Download one hour of tick data from Dukascopy. month0 is 0-indexed (Jan=0).
    Returns a list of {ts_ms, bid, ask} or None on failure / empty hour."""
    url = (f"{DUKASCOPY_BASE}/{dk_sym}/"
           f"{year:04d}/{month0:02d}/{day:02d}/{hour:02d}h_ticks.bi5")
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "ron-dukascopy/1.0"})
    except Exception:
        return None
    if resp.status_code != 200 or not resp.content:
        return None
    try:
        decompressed = lzma.decompress(resp.content)
    except Exception:
        # Empty hours are sometimes 0 bytes which lzma can't decode — skip silently
        return None

    pf = 10 ** point_factor
    hour_start_ms = int(datetime(year, month0 + 1, day, hour,
                                  tzinfo=timezone.utc).timestamp() * 1000)
    record_size = 20
    ticks = []
    for i in range(0, len(decompressed), record_size):
        if i + record_size > len(decompressed):
            break
        # Big-endian: uint32 ms_offset, uint32 ask_raw, uint32 bid_raw, float32 ask_vol, float32 bid_vol
        try:
            offset_ms, ask_raw, bid_raw, _ask_vol, _bid_vol = struct.unpack(
                ">IIIff", decompressed[i:i + record_size])
        except struct.error:
            continue
        ticks.append({
            "ts_ms": hour_start_ms + offset_ms,
            "bid":   bid_raw / pf,
            "ask":   ask_raw / pf,
        })
    return ticks if ticks else None


def _ticks_to_1m_candles(ticks: list, symbol: str) -> list:
    """Aggregate tick stream to 1m OHLCV using mid-price."""
    if not ticks:
        return []
    df = pd.DataFrame(ticks)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    ohlc = df["mid"].resample("1min").ohlc()
    vol  = df["mid"].resample("1min").count().rename("volume")
    out = pd.concat([ohlc, vol], axis=1).dropna(subset=["open"])
    candles = []
    for ts, row in out.iterrows():
        # Match parse_dukascopy_csv format: naive datetime string + 5dp rounding
        candles.append({
            "symbol":    symbol,
            "timeframe": "1m",
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open":      round(float(row["open"]), 5),
            "high":      round(float(row["high"]), 5),
            "low":       round(float(row["low"]), 5),
            "close":     round(float(row["close"]), 5),
            "volume":    int(row["volume"]),
        })
    return candles


def _store_candles_with_diagnostics(candles: list) -> dict:
    """Same as store_candles_in_supabase but returns first error details for debugging."""
    if not candles:
        return {"stored": 0, "first_error": None, "batches": 0}

    stored = 0
    first_error = None
    batches = 0
    for i in range(0, len(candles), 100):
        batch = candles[i:i+100]
        batches += 1
        url = f"{SUPABASE_URL}/rest/v1/candle_history"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=ignore-duplicates,return=minimal"
        }
        try:
            resp = requests.post(url, headers=headers, json=batch, timeout=30)
            if resp.status_code in (200, 201):
                stored += len(batch)
            else:
                if first_error is None:
                    first_error = {
                        "status": resp.status_code,
                        "body":   (resp.text or "")[:500],
                        "first_row": batch[0] if batch else None,
                    }
                logger.error(f"Supabase insert error: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            if first_error is None:
                first_error = {"exception": str(e), "first_row": batch[0] if batch else None}
    return {"stored": stored, "first_error": first_error, "batches": batches}


@app.route("/ingest/dukascopy-direct", methods=["POST"])
def ingest_dukascopy_direct():
    """Fetch 1m candles directly from Dukascopy's public datafeed and store
    them in candle_history. No manual CSV uploads.

    Body: {
        "symbol":   "XAUUSD",          # standardized (see DUKASCOPY_INSTRUMENT_MAP)
        "start":    "2022-01-01",      # inclusive UTC date
        "end":      "2022-01-14",      # inclusive
        "max_days": 31,                # safety cap, default 31
        "skip_weekends": true          # default true
    }

    Note: Render's HTTP timeout limits how big a range can fit in one call.
    For multi-month back-fills, call this endpoint once per chunk (~2 weeks).
    """
    api_key = request.headers.get("X-API-Key", "")
    if api_key != os.environ.get("RON_API_KEY", "gainedge-ron-2026"):
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json() or {}
    symbol         = str(body.get("symbol", "XAUUSD")).upper()
    start          = body.get("start")
    end            = body.get("end")
    max_days       = int(body.get("max_days", 31))
    skip_weekends  = bool(body.get("skip_weekends", True))
    max_workers    = int(body.get("max_workers", 8))

    if not start or not end:
        return jsonify({"error": "start and end required (YYYY-MM-DD)"}), 400
    if symbol not in DUKASCOPY_INSTRUMENT_MAP:
        return jsonify({
            "error":     "unsupported symbol",
            "supported": list(DUKASCOPY_INSTRUMENT_MAP.keys()),
        }), 400

    dk_sym, pf = DUKASCOPY_INSTRUMENT_MAP[symbol]
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        return jsonify({"error": f"bad date format: {e}"}), 400

    days = (end_dt - start_dt).days + 1
    if days < 1:
        return jsonify({"error": "end must be >= start"}), 400
    if days > max_days:
        return jsonify({
            "error":    f"range too large: {days} days, max_days={max_days}",
            "hint":     "split into ~14-day chunks; call /ingest/dukascopy-direct multiple times",
        }), 400

    # Build the list of hourly fetches
    hours = []
    cur = start_dt
    while cur <= end_dt:
        if skip_weekends and cur.weekday() >= 5:
            cur += timedelta(days=1)
            continue
        for h in range(24):
            hours.append((cur.year, cur.month - 1, cur.day, h))
        cur += timedelta(days=1)

    logger.info(f"Dukascopy ingest {symbol}: {days} days → {len(hours)} hourly fetches")

    all_ticks = []
    failed = 0

    def worker(args):
        y, m0, d, h = args
        return _fetch_dukascopy_hour(dk_sym, pf, y, m0, d, h)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ticks in pool.map(worker, hours):
            if ticks is None:
                failed += 1
                continue
            all_ticks.extend(ticks)

    if not all_ticks:
        return jsonify({
            "error":            "no tick data downloaded",
            "hours_requested":  len(hours),
            "hours_failed":     failed,
            "hint":             "check date range; weekends and market holidays return no data",
        }), 502

    all_ticks.sort(key=lambda t: t["ts_ms"])
    candles = _ticks_to_1m_candles(all_ticks, symbol)
    insert_result = _store_candles_with_diagnostics(candles)

    return jsonify({
        "status":             "complete",
        "symbol":             symbol,
        "start":              start,
        "end":                end,
        "days":               days,
        "hours_requested":    len(hours),
        "hours_failed":       failed,
        "ticks_downloaded":   len(all_ticks),
        "candles_aggregated": len(candles),
        "candles_stored":     insert_result["stored"],
        "insert_batches":     insert_result["batches"],
        "first_insert_error": insert_result["first_error"],
    })


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"RON ML Server starting on port {port}")

    # Try to load existing model on startup
    model = load_model()
    if model:
        logger.info("Existing model loaded successfully")
    else:
        logger.info("No existing model found — will train when /train is called")

    app.run(host="0.0.0.0", port=port)

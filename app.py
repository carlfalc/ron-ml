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
from datetime import datetime, timedelta
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
    """Bulk insert candles into candle_history table."""
    if not candles:
        return 0

    stored = 0
    # Insert in batches of 100
    for i in range(0, len(candles), 100):
        batch = candles[i:i+100]
        url = f"{SUPABASE_URL}/rest/v1/candle_history"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=ignore-duplicates,return=minimal"
        }
        resp = requests.post(url, headers=headers, json=batch)
        if resp.status_code in (200, 201):
            stored += len(batch)
        else:
            logger.error(f"Supabase insert error: {resp.status_code} {resp.text}")

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
    "XAU-USD": "XAUUSD",
    "XAU-AUD": "XAUAUD",
    "XAG-USD": "XAGUSD",
    "USD-CAD": "USDCAD",
    "USD-JPY": "USDJPY",
    "USA500IDX-USD": "US500",
    "NZD-USD": "NZDUSD",
    "AUD-USD": "AUDUSD",
    "EUR-USD": "EURUSD",
    "GBP-USD": "GBPUSD",
    "GBP-JPY": "GBPJPY",
    "EUR-JPY": "EURJPY",
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

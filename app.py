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

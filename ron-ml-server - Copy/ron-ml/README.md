# RON AI - Machine Learning Trading Intelligence

## Overview
RON ML is the machine learning engine for GAINEDGE's trading intelligence platform. It uses XGBoost to predict trade outcomes based on market conditions, learning from ALL GAINEDGE users' trading data.

## What RON ML Does
1. **Trains** on signal_outcomes data from all GAINEDGE users
2. **Predicts** win probability for new trading setups
3. **Analyses** setups by combining ML prediction with historical pattern stats
4. **Improves** automatically as more data accumulates

## API Endpoints

### GET /health
Health check — returns model status.

### POST /train
Trigger model retraining. Requires `X-API-Key` header.
Call weekly via cron job or manually.

### POST /predict
Predict win probability for a trade setup.
```json
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
```

Response:
```json
{
  "probability": 0.73,
  "confidence_label": "HIGH CONVICTION",
  "recommendation": "TAKE TRADE",
  "model_available": true
}
```

### POST /analyse-setup
Full analysis combining ML + historical stats + reasoning.
Same input as /predict but returns comprehensive analysis with
human-readable reasoning text for Ask RON responses.

### GET /feature-importance
Returns which factors matter most to the model.

### GET /model-stats
Returns current model performance metrics.

## Deployment to Render.com

### Option 1: One-Click Deploy
1. Push this code to a GitHub repo (e.g., `carlfalc/ron-ml`)
2. Go to https://render.com
3. Click "New" → "Web Service"
4. Connect your GitHub repo
5. Render auto-detects the `render.yaml` config
6. Add environment variables:
   - `SUPABASE_URL` = https://ecsztqtyttnqdnsphxip.supabase.co
   - `SUPABASE_SERVICE_KEY` = (your service role key)
   - `RON_API_KEY` = (create a secure random string)
7. Click "Deploy"

### Option 2: Manual Setup
1. Create a new Web Service on Render
2. Connect to your GitHub repo
3. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
4. Add environment variables as above
5. Deploy

### Cost
~$7/month on Render Starter plan (512MB RAM, shared CPU)
Sufficient for training on 10,000+ signals and serving predictions.

## Connecting to GAINEDGE

Once deployed, add the Render URL as a secret in your Lovable/Supabase project:
- `RON_ML_URL` = https://ron-ml.onrender.com (or whatever Render assigns)
- `RON_ML_API_KEY` = (same key you set on Render)

Then update the compute-market-data edge function to call RON ML
before generating signals:

```javascript
// In compute-market-data edge function:
const ronResponse = await fetch(`${RON_ML_URL}/predict`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": RON_ML_API_KEY
  },
  body: JSON.stringify({
    symbol, direction, adx_at_entry, rsi_at_entry,
    stoch_rsi_at_entry, macd_status, confidence,
    session, hour_utc, day_of_week, pattern_active
  })
});
const { probability, recommendation } = await ronResponse.json();

// Only generate signal if RON ML says probability >= 0.60
if (probability >= 0.60) {
  // Insert signal with ML-boosted confidence
}
```

## Weekly Retraining
Set up a cron job (via Render Cron or external service) to call:
```
POST https://ron-ml.onrender.com/train
Headers: X-API-Key: your-api-key
```
Run every Sunday at 00:00 UTC to retrain on the latest week's data.

## Model Details
- **Algorithm**: XGBoost (Gradient Boosted Decision Trees)
- **Features**: 29 input features covering indicators, session, patterns, instruments
- **Training**: 80/20 train/test split, binary classification (WIN vs LOSS)
- **Min data**: 50 resolved signals required to train
- **Retraining**: Weekly automatic, or on-demand via API

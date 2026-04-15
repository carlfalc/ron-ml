# Render.com deployment configuration
# Deploy as a Web Service with the following settings:
#   Build Command: pip install -r requirements.txt
#   Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120

# Environment variables to set on Render:
#   SUPABASE_URL = https://ecsztqtyttnqdnsphxip.supabase.co
#   SUPABASE_SERVICE_KEY = (your Supabase service role key)
#   RON_API_KEY = (create a secure random key for auth)
#   PORT = 10000 (Render default)

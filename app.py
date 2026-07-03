from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import sqlite3
import json
from datetime import datetime

app = Flask(__name__)

@app.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

# Load models
fraud_model = joblib.load('models/fraud_model.pkl')
sms_model   = joblib.load('models/sms_model.pkl')
vectorizer  = joblib.load('models/vectorizer.pkl')
le_txn      = joblib.load('models/le_txn.pkl')
le_merchant = joblib.load('models/le_merchant.pkl')

try:
    with open('models/metrics.json') as f:
        metrics = json.load(f)
    FRAUD_ACC = float(metrics.get('fraud_accuracy', 0))
    SMS_ACC   = float(metrics.get('sms_accuracy', 0))
    if not (0 <= FRAUD_ACC <= 100): FRAUD_ACC = 0.0
    if not (0 <= SMS_ACC   <= 100): SMS_ACC   = 0.0
except Exception:
    FRAUD_ACC = 0.0
    SMS_ACC   = 0.0

KNOWN_MERCHANTS = list(le_merchant.classes_)
KNOWN_TXN_TYPES = list(le_txn.classes_)

MAX_AMOUNT      = 10_000_000   # Rs. 1 crore max
MAX_MERCHANT_LEN = 100
MAX_SMS_LEN      = 1000

def init_db():
    conn = sqlite3.connect('scamsense.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS transactions (
                 id         INTEGER PRIMARY KEY AUTOINCREMENT,
                 amount     REAL,
                 txn_type   TEXT,
                 merchant   TEXT,
                 result     TEXT,
                 confidence REAL,
                 timestamp  TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sms_checks (
                 id        INTEGER PRIMARY KEY AUTOINCREMENT,
                 message   TEXT,
                 result    TEXT,
                 timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html', fraud_acc=FRAUD_ACC, sms_acc=SMS_ACC)

@app.route('/transaction')
def transaction():
    return render_template('transaction.html')

@app.route('/sms')
def sms():
    return render_template('sms.html')

@app.route('/about')
def about():
    return render_template('about.html', fraud_acc=FRAUD_ACC, sms_acc=SMS_ACC)

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('scamsense.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM transactions")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM transactions WHERE result='FRAUD'")
    frauds = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM sms_checks WHERE result='PHISHING'")
    phishing = c.fetchone()[0]
    c.execute("""SELECT id, amount, txn_type, merchant, result, confidence, timestamp
                 FROM transactions ORDER BY id DESC LIMIT 10""")
    rows = c.fetchall()
    conn.close()

    # Sanitize rows before sending to template
    recent = []
    for row in rows:
        rid, amount, txn_type, merchant, result, confidence, timestamp = row
        try:
            amount = float(amount) if amount is not None else 0.0
        except Exception:
            amount = 0.0
        try:
            confidence = float(confidence)
            if not (0 <= confidence <= 100):
                confidence = 0.0
        except Exception:
            confidence = 0.0
        merchant  = (str(merchant)[:50] + '...') if merchant and len(str(merchant)) > 50 else (merchant or 'N/A')
        result    = result if result in ('FRAUD', 'LEGITIMATE') else 'UNKNOWN'
        timestamp = timestamp if timestamp else 'N/A'
        recent.append((rid, amount, txn_type, merchant, result, confidence, timestamp))

    return render_template('dashboard.html',
                           total=total, frauds=frauds,
                           phishing=phishing, recent=recent,
                           fraud_acc=FRAUD_ACC, sms_acc=SMS_ACC)

@app.route('/predict_transaction', methods=['POST'])
def predict_transaction():
    try:
        # FIX 1: Validate JSON body
        if not request.is_json or request.json is None:
            return jsonify({'error': 'Invalid request. JSON body required.'}), 400

        data = request.json

        # FIX 2: Check required fields
        if 'amount' not in data:
            return jsonify({'error': 'Amount is required.'}), 400

        # FIX 3: Validate amount range
        try:
            amount = float(data['amount'])
        except (ValueError, TypeError):
            return jsonify({'error': 'Amount must be a number.'}), 400

        if amount <= 0:
            return jsonify({'error': 'Amount must be greater than 0.'}), 400
        if amount > MAX_AMOUNT:
            return jsonify({'error': f'Amount cannot exceed Rs.{MAX_AMOUNT:,}.'}), 400

        # FIX 4: Validate merchant length
        merchant = str(data.get('merchant', '')).strip()
        if len(merchant) > MAX_MERCHANT_LEN:
            return jsonify({'error': f'Merchant name too long. Max {MAX_MERCHANT_LEN} characters.'}), 400
        if not merchant:
            merchant = 'Unknown'

        txn_type = str(data.get('txn_type', 'UPI Transfer')).strip()
        if txn_type not in KNOWN_TXN_TYPES:
            txn_type = KNOWN_TXN_TYPES[0]

        # Encode features
        suspicious_keywords = [
            'unknown', 'verify', 'claim', 'prize', 'lottery',
            'free', 'urgent', 'win', 'lucky', 'fake', 'secure',
            'alert', 'block', 'suspend', 'kyc', 'reward', 'gift'
        ]
        merchant_lower = merchant.lower()
        is_suspicious  = any(kw in merchant_lower for kw in suspicious_keywords)

        txn_enc      = le_txn.transform([txn_type])[0]
        merchant_enc = le_merchant.transform(['Unknown'])[0] \
                       if (is_suspicious or merchant.title() not in KNOWN_MERCHANTS) \
                       else le_merchant.transform([merchant.title()])[0]

        # FIX 5: Use actual transaction time from server (honest — document this)
        hour          = datetime.now().hour
        is_round      = int(amount % 1000 == 0 and amount > 5000)
        is_high       = int(amount > 50000)
        is_late_night = int(0 <= hour < 5)

        features = pd.DataFrame([{
            'amount':          amount,
            'txn_type':        txn_enc,
            'hour_of_day':     hour,
            'merchant':        merchant_enc,
            'is_round_amount': is_round,
            'is_high_amount':  is_high,
            'is_late_night':   is_late_night
        }])

        # FIX 7: Pure ML prediction — no manual override
        # Suspicious keyword merchant is mapped to 'Unknown' merchant_enc
        # which the model already learned is high-risk. No manual override needed.
        prediction  = fraud_model.predict(features)[0]
        probability = float(fraud_model.predict_proba(features)[0][1])

        result     = 'FRAUD' if prediction == 1 else 'LEGITIMATE'
        confidence = round(probability * 100, 2)

        conn = sqlite3.connect('scamsense.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO transactions (amount, txn_type, merchant, result, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (amount, txn_type, merchant, result, confidence,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()

        return jsonify({'result': result, 'confidence': confidence})

    except Exception:
        # FIX 5: Never expose internal errors
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500


@app.route('/predict_sms', methods=['POST'])
def predict_sms():
    try:
        # FIX 6: Validate JSON and required fields
        if not request.is_json or request.json is None:
            return jsonify({'error': 'Invalid request. JSON body required.'}), 400

        data = request.json

        if 'message' not in data:
            return jsonify({'error': 'Message field is required.'}), 400

        message = str(data['message']).strip()

        if not message:
            return jsonify({'error': 'Message cannot be empty.'}), 400

        if len(message) > MAX_SMS_LEN:
            return jsonify({'error': f'Message too long. Max {MAX_SMS_LEN} characters.'}), 400

        vec        = vectorizer.transform([message])
        prediction = sms_model.predict(vec)[0]
        result     = 'PHISHING' if prediction == 1 else 'SAFE'

        conn = sqlite3.connect('scamsense.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO sms_checks (message, result, timestamp) VALUES (?, ?, ?)",
            (message, result, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()

        return jsonify({'result': result})

    except Exception:
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('404.html'), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

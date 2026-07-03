import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

os.makedirs('models', exist_ok=True)

# ── FRAUD MODEL WITH REAL FEATURES ───────────────────────────────
print("🔄 Building fraud dataset with real features...")

np.random.seed(42)
n = 5000

# Simulate realistic transaction data
amounts = np.concatenate([
    np.random.exponential(500, 4500),   # normal transactions
    np.random.uniform(10000, 100000, 500)  # large suspicious ones
])

txn_types = np.random.choice(
    ['UPI Transfer', 'Credit Card', 'Debit Card', 'Net Banking'],
    n, p=[0.5, 0.2, 0.2, 0.1]
)

hours = np.concatenate([
    np.random.randint(8, 22, 4500),    # normal hours
    np.random.randint(0, 5, 500)       # suspicious late night
])

merchants = np.random.choice(
    ['Amazon', 'Swiggy', 'Zomato', 'Flipkart', 'Unknown',
     'Verify Now', 'Claim Prize', 'BigBasket', 'PhonePe', 'Paytm'],
    n, p=[0.15, 0.15, 0.1, 0.15, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05]
)

# Simulate fraud label based on real patterns
labels = []
for i in range(n):
    fraud_prob = 0.02  # base 2% fraud rate
    if amounts[i] > 50000: fraud_prob += 0.4
    if hours[i] < 5: fraud_prob += 0.3
    if merchants[i] in ['Unknown', 'Verify Now', 'Claim Prize']: fraud_prob += 0.5
    if amounts[i] % 1000 == 0 and amounts[i] > 10000: fraud_prob += 0.1
    labels.append(1 if np.random.random() < min(fraud_prob, 0.95) else 0)

# Encode categorical features
le_txn = LabelEncoder()
le_merchant = LabelEncoder()

txn_encoded = le_txn.fit_transform(txn_types)
merchant_encoded = le_merchant.fit_transform(merchants)

df = pd.DataFrame({
    'amount': amounts,
    'txn_type': txn_encoded,
    'hour_of_day': hours,
    'merchant': merchant_encoded,
    'is_round_amount': (amounts % 1000 == 0).astype(int),
    'is_high_amount': (amounts > 50000).astype(int),
    'is_late_night': ((hours >= 0) & (hours < 5)).astype(int),
    'label': labels
})

print(f"Dataset: {len(df)} transactions | Fraud: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

fraud_model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
fraud_model.fit(X_train, y_train)

y_pred = fraud_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Fraud Model Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# ── SMS MODEL WITH REAL DATASET ───────────────────────────────────
print("\n🔄 Building SMS phishing model with larger dataset...")

# 100 realistic SMS samples (balanced)
sms_samples = [
    # PHISHING (label=1)
    ("URGENT: Your SBI account blocked! Click http://sbi-verify.xyz to reactivate", 1),
    ("Congratulations! You won Rs.50,000 lottery. Call 9999999999 to claim now!", 1),
    ("FREE iPhone 15! You are selected. Click http://win-prize.com immediately", 1),
    ("Your KYC expired. Update now at http://fake-kyc.com or account suspended", 1),
    ("ALERT: Unusual login to your account. Verify at http://hdfc-secure.net", 1),
    ("You have won Rs.1,00,000 in lucky draw! Send bank details to claim", 1),
    ("Your UPI PIN expired. Reset at http://upi-reset.com within 24 hours", 1),
    ("Get Rs.999 free recharge! Limited offer. Click http://free-recharge.xyz", 1),
    ("WINNER! Claim your gift voucher worth Rs.5000. Call 8888888888 now!", 1),
    ("Your PAN card is blocked. Verify at http://pan-verify.com immediately", 1),
    ("Earn Rs.10,000 daily from home! Guaranteed. WhatsApp 7777777777", 1),
    ("Your electricity will be cut in 2 hours. Pay now at http://fake-bescom.com", 1),
    ("Income Tax refund of Rs.8,500 pending. Submit details at http://it-refund.xyz", 1),
    ("BANK ALERT: Rs.49,999 debited. If not you, click http://freeze-account.com", 1),
    ("You are pre-approved for Rs.5 lakh loan. No documents. Call 9876543210", 1),
    ("Dear user, your Paytm wallet suspended. KYC update: http://paytm-kyc.net", 1),
    ("Congratulations! BSNL customer selected for Rs.25,000 reward. Click here", 1),
    ("Your Amazon order refund Rs.2,340 failed. Retry at http://amzn-refund.xyz", 1),
    ("IRCTC: Your account hacked! Secure now at http://irctc-secure.com", 1),
    ("Get rich quick! Invest Rs.500 get Rs.5000 in 7 days. Guaranteed returns!", 1),
    ("OTP for unknown transaction: 483920. Share with our agent to cancel", 1),
    ("Your Aadhaar linked to suspicious activity. Call 1800-FAKE to resolve", 1),
    ("Win free trip to Dubai! You are selected. Register at http://trip-win.com", 1),
    ("FINAL WARNING: Pay Rs.500 or lose your sim card today!", 1),
    ("Job offer: Work from home Rs.50,000/month. No experience. Whatsapp now", 1),
    ("Your credit card limit increased to Rs.5L. Activate at http://card-limit.xyz", 1),
    ("PM Kisan Yojana Rs.6000 ready. Submit Aadhaar at http://pmkisan-fake.com", 1),
    ("Dear customer, you have a pending cashback of Rs.1,299. Claim: http://cb.xyz", 1),
    ("Security breach on your account! Immediate action: http://secure-now.net", 1),
    ("Investment opportunity: 30% monthly returns guaranteed. Call 9988776655", 1),
    ("Your Jio SIM will deactivate in 24hrs. Update KYC: http://jio-kyc.fake", 1),
    ("Ration card benefits of Rs.2000 waiting. Submit details at http://ration.xyz", 1),
    ("HDFC: Rs.1,00,000 debited from your account. Dispute at http://hdfc-fake.net", 1),
    ("Your driving licence renewal pending. Pay Rs.200 at http://rto-fake.com", 1),
    ("Free COVID insurance worth Rs.50,000! Register at http://covid-insure.xyz", 1),
    ("You won a Samsung TV! Collect by paying Rs.500 shipping. Call 9876512345", 1),
    ("Your EPF withdrawal request pending approval. Submit at http://epf-fake.com", 1),
    ("URGENT: Your demat account has suspicious trades. Call 1800-BROKER-FAKE", 1),
    ("Earn by liking YouTube videos Rs.500/hour. Join: http://earn-online.xyz", 1),
    ("Gas subsidy of Rs.1,600 pending. Link bank at http://lpg-subsidy-fake.com", 1),
    ("Fake transaction of Rs.75,000 detected. Block card: http://block-now.xyz", 1),
    ("Your voter ID has an error. Correct at http://eci-fake.com before election", 1),
    ("Scheme: Invest Rs.1000 today and get Rs.10,000 in 30 days. Guaranteed!", 1),
    ("Dear winner, your number selected in KBC. Call 00447782341234 to claim", 1),
    ("Your Netflix subscription cancelled. Reactivate at http://netflix-fake.net", 1),
    ("IRCTC refund of Rs.1,240 initiated. Provide account details to receive", 1),
    ("Suspicious login from Delhi. If not you verify: http://gmail-secure-fake.com", 1),
    ("PM Awas Yojana house allotted! Submit docs at http://pmay-register.xyz", 1),
    ("Your FD matured Rs.2,50,000. Claim by calling 9988001122 before it expires", 1),
    ("Work from home data entry job Rs.800/hour. No investment. Join now!", 1),
    # LEGITIMATE (label=0)
    ("Your OTP for SBI login is 847291. Valid for 10 minutes. Do not share.", 0),
    ("Dear customer, your UPI txn of Rs.500 to Swiggy is successful. Ref:TXN123", 0),
    ("Your Flipkart order #FL789 has been shipped. Expected delivery: Tomorrow", 0),
    ("HDFC Bank: Your salary of Rs.45,000 credited on 01-Jul. Avl bal: Rs.52,340", 0),
    ("Your Zomato order is out for delivery. Estimated time: 15 minutes", 0),
    ("Amazon: Your return for order #AMZ456 has been approved. Refund in 5-7 days", 0),
    ("Dear Angel, your EMI of Rs.3,200 due on 5th Jul. Auto-debit enabled.", 0),
    ("Your electricity bill of Rs.1,840 paid successfully. Receipt: BESCOM-2026", 0),
    ("Reminder: Your doctor appointment is tomorrow at 10:30 AM at Apollo Hospital", 0),
    ("Hi, are you free this evening? Let's catch up for dinner.", 0),
    ("Your PF balance is Rs.1,25,430. Last contribution: Rs.1,800 on Jun 2026", 0),
    ("IRCTC booking confirmed. PNR: 1234567890. Train: 12345 on 05-Jul-2026", 0),
    ("Your Airtel bill of Rs.399 is due on 10th Jul. Pay via MyAirtel app.", 0),
    ("Meeting rescheduled to 3 PM tomorrow. Please confirm your attendance.", 0),
    ("Your blood test reports are ready. Please collect from lab counter 3.", 0),
    ("Paytm: Rs.200 added to your wallet. New balance: Rs.450", 0),
    ("Your driving test is scheduled for 15-Jul-2026 at 9 AM at RTO Bengaluru.", 0),
    ("NEFT of Rs.10,000 to ICICI A/C XX1234 successful. UTR: HDFC2026070112345", 0),
    ("Your Jio data pack of 1.5GB/day activated. Valid till 28-Jul-2026.", 0),
    ("Class 12 result declared. Check at official board website.", 0),
    ("Your LPG cylinder booking confirmed. Expected delivery: 3-5 working days", 0),
    ("Hi team, tomorrow's standup moved to 10 AM. Please join on time.", 0),
    ("Your Uber ride receipt: Rs.180 from MG Road to Whitefield on 01-Jul", 0),
    ("Aadhaar update successful. New mobile number linked: XXXXXXX890", 0),
    ("Your mutual fund SIP of Rs.2,000 deducted successfully on 1st Jul 2026", 0),
    ("BESCOM: Scheduled maintenance on 3rd Jul 6AM-10AM. Power outage expected.", 0),
    ("Your Amazon Prime subscription renewed for Rs.1,499. Next renewal: Jul 2027", 0),
    ("College exam timetable released. Check your college portal for details.", 0),
    ("Your Swiggy One membership renewed. Enjoy free delivery for next 3 months.", 0),
    ("PhonePe: UPI payment of Rs.1,200 to BigBasket successful. Order confirmed.", 0),
    ("Your vehicle insurance renewed. Policy valid till 01-Jul-2027. Drive safe!", 0),
    ("Income tax return filed successfully. Acknowledgement: ITR2026XXXXX", 0),
    ("Your CRED coins: 2,450. Use them to pay your credit card bill this month.", 0),
    ("Reminder: Parent-teacher meeting on Saturday 10 AM at school auditorium.", 0),
    ("Your health insurance claim of Rs.12,000 approved. Amount credited in 2 days", 0),
    ("Zepto order delivered! Rate your experience to help us improve.", 0),
    ("Your passport application status: Under Police Verification.", 0),
    ("SBI: Fixed deposit of Rs.50,000 matured. Amount credited to your account.", 0),
    ("Your Github pull request #42 has been merged by the reviewer.", 0),
    ("NEET 2026 admit card available. Download from official NTA website.", 0),
    ("Your Google account was signed in from Chrome on Windows. Was this you?", 0),
    ("Ola: Your ride from Airport to home completed. Total: Rs.420. Tip driver?", 0),
    ("Your blood donation camp certificate is ready. Collect from centre.", 0),
    ("Wi-Fi bill of Rs.699 paid. Thank you for using ACT Fibernet.", 0),
    ("Your job application at TCS has moved to the next round. Check email.", 0),
    ("BSNL: Your broadband speed upgraded to 100Mbps at no extra cost.", 0),
    ("Your college fees receipt for Semester 5 has been generated.", 0),
    ("Reminder: Submit your project report by Friday 5 PM to the department.", 0),
    ("Your car service at Maruti Service Center is complete. Ready for pickup.", 0),
    ("Happy Birthday! Wishing you a wonderful day filled with joy and success.", 0),
]

sms_df = pd.DataFrame(sms_samples, columns=['message', 'label'])
print(f"SMS Dataset: {len(sms_df)} messages | Spam: {sms_df['label'].sum()} | Ham: {(sms_df['label']==0).sum()}")

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X_sms = vectorizer.fit_transform(sms_df['message'])
y_sms = sms_df['label']

X_sms_train, X_sms_test, y_sms_train, y_sms_test = train_test_split(
    X_sms, y_sms, test_size=0.2, random_state=42, stratify=y_sms)

nlp_model = LogisticRegression(max_iter=1000, class_weight='balanced')
nlp_model.fit(X_sms_train, y_sms_train)

sms_pred = nlp_model.predict(X_sms_test)
sms_acc = accuracy_score(y_sms_test, sms_pred)
print(f"✅ SMS Model Accuracy: {sms_acc*100:.2f}%")
print(classification_report(y_sms_test, sms_pred, target_names=['Legitimate', 'Phishing']))

# Save models and encoders
joblib.dump(fraud_model, 'models/fraud_model.pkl')
joblib.dump(nlp_model,   'models/sms_model.pkl')
joblib.dump(vectorizer,  'models/vectorizer.pkl')
joblib.dump(le_txn,      'models/le_txn.pkl')
joblib.dump(le_merchant, 'models/le_merchant.pkl')

# Save actual accuracy values for use in app
import json
metrics = {
    'fraud_accuracy': round(acc * 100, 2),
    'sms_accuracy': round(sms_acc * 100, 2)
}
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f)

print("\n✅ All models saved!")
print(f"📊 Fraud Accuracy: {metrics['fraud_accuracy']}%")
print(f"📊 SMS Accuracy:   {metrics['sms_accuracy']}%")
print("🚀 Now run: python app.py")
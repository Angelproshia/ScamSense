# ScamSense — AI-Powered UPI & Digital Payment Fraud Detection

A final year project that uses Machine Learning and NLP to detect
fraudulent transactions and phishing SMS messages in real time.

## Features
- Transaction fraud detection using Random Forest (83.9% accuracy)
- SMS phishing detection using TF-IDF + Logistic Regression (95% accuracy)
- Live dashboard with detection history
- Input validation and secure error handling

## Tech Stack
Python, Flask, Scikit-learn, Pandas, SQLite, Bootstrap 5

## Setup Instructions

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train models
python train_model.py

### 3. Run the app
python app.py

### 4. Open browser
http://127.0.0.1:5000

## Project Structure
- app.py          — Flask backend
- train_model.py  — ML model training
- templates/      — HTML pages
- static/         — CSS and JS
- models/         — Saved ML models

## Limitations
- Fraud model trained on simulated data
- SMS dataset contains 100 labeled samples
- No real-time bank API integration
- Time used is server submission time, not actual transaction time

## Author
Angel Proshia | 2026

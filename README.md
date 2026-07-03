# ScamSense — AI-Powered UPI Fraud Detection

AI-powered system that detects fraudulent UPI transactions 
and SMS phishing attacks in real time using Machine Learning.

## Live Demo
🌐 Coming soon on Render.com

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
- app.py           Flask backend and API routes
- train_model.py   ML model training script
- templates/       HTML pages
- static/          CSS and JS files
- models/          Saved ML models (auto-generated)

## Limitations
- Fraud model trained on simulated data
- SMS dataset contains 100 labeled samples
- No real-time bank API integration

## Author
Angel Proshia | 2026

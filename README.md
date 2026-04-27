# Churn Intelligence — Explainable Customer Churn Prediction Platform

![Python](https://img.shields.io/badge/Python-ML-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end churn prediction platform that predicts telecom customer churn, explains *why* using SHAP, and recommends retention actions through a production-style analytics dashboard.

## Highlights
- XGBoost model with SMOTE + hyperparameter tuning
- Calibrated decision threshold (not default 0.5)
- SHAP-based explainability
- Business-context retention recommendations
- FastAPI backend with modular architecture
- Monitoring + drift detection
- SaaS-style frontend with risk tiers, confidence scoring, and ROI estimates

## Demo Features
- Predict churn risk in real time
- View top drivers behind each prediction
- Get retention recommendations
- Monitor model behavior and prediction drift

## Architecture
Data → Preprocessing → Feature Engineering → XGBoost → Threshold Calibration → SHAP Explanations → FastAPI → Dashboard

## Model Performance
- AUC-ROC: ~0.845  
- F1 Score: ~0.62  
- Optimal Threshold: ~0.40

## Tech Stack
- Python
- FastAPI
- XGBoost
- SHAP
- Pydantic v2
- HTML/CSS/JS Dashboard

## Project Structure
```text
churn_project/
├── data/
├── training/
├── backend/
│   ├── routers/
│   ├── services/
│   ├── models/
│   ├── utils/
│   └── model/
└── frontend/
```

## Quick Start
### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train model
```bash
python training/train.py
```

### 3. Run API
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open dashboard
Open:
frontend/index.html

## API Endpoints
- POST /predict
- POST /batch_predict
- GET /overview
- GET /segments
- GET /churn_trend
- GET /model/info
- GET /monitoring/stats

## Example Prediction Output
```json
{
 "churn": 1,
 "probability": 0.8342,
 "confidence":"High",
 "action":"Trigger immediate retention intervention"
}
```

## Screenshots
(Add images here)
- Dashboard overview
- SHAP explanation panel
- Risk tier cards
- Monitoring view

## Why This Project Is Different
Unlike basic churn classifiers, this project includes:

✅ Explainable AI  
✅ Threshold calibration  
✅ Business action recommendations  
✅ Drift monitoring  
✅ Production API structure

## Future Improvements
- Deploy with Docker
- Add authentication
- Real-time streaming predictions
- Multi-industry churn support

## Author
Athithya S

LinkedIn:
www.linkedin.com/in/athithya-s-b91b4b292

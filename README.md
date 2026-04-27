# Churn Intelligence Platform v2.0

Production-grade customer churn prediction system built on the Telco dataset.  
XGBoost + SMOTE + SHAP + FastAPI + a dark-themed SaaS dashboard.

---

## Project Structure

```text
churn_project/
├── data/
│   └── telco.csv
├── training/
│   └── train.py
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── model/
│   │   ├── churn_model.pkl
│   │   ├── scaler.pkl
│   │   ├── feature_columns.pkl
│   │   ├── threshold.pkl
│   │   └── model_meta.json
│   ├── routers/
│   │   ├── predict.py
│   │   ├── customers.py
│   │   ├── analytics.py
│   │   └── monitoring.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── ml_service.py
│   │   └── data_service.py
│   └── utils/
│       ├── preprocessing.py
│       ├── explain.py
│       ├── insights.py
│       └── confidence.py
└── frontend/
    └── index.html
```

---

## Quick Start

### 1. Place the dataset

```text
churn_project/data/telco.csv
```

### 2. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Train the model

```bash
python training/train.py
```

Expected output:

```text
AUC-ROC:  ~0.845
F1 Score: ~0.62
Optimal threshold: ~0.38–0.44
```

### 4. Start the API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Open the dashboard

Open `frontend/index.html`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /predict | Single churn prediction (18 feats) |
| POST | /batch_predict | Batch predict up to 200 customers |
| GET | /customers | Full customer list from CSV |
| GET | /overview | Summary stats |
| GET | /segments | Contract-based segments |
| GET | /churn_trend | Churn rate by tenure bracket |
| GET | /model/info | Model metadata + AUC/F1 |
| GET | /monitoring/stats | Prediction log + drift detection |
| DELETE | /monitoring/reset | Clear prediction log |

---

## Key Improvements Over v1

| Area | v1 | v2 |
|------|----|----|
| Model | RandomForest | XGBoost + SMOTE + tuning |
| Threshold | 0.5 | Calibrated (~0.40) |
| Preprocessing | Basic | Full pipeline |
| Feature Eng. | None | 10 derived features |
| Insights | Generic | Business-context rules |
| Confidence | Threshold-based | SHAP + boundary distance |
| API | Flat main.py | Routers + services |
| Monitoring | None | Drift detection |
| Frontend | Charts only | Risk tiers + ROI |

---

## Prediction Request Example

```json
{
  "tenure": 3,
  "MonthlyCharges": 79.85,
  "TotalCharges": 239.55
}
```

```json
{
  "churn": 1,
  "probability": 0.8342
}
```

"""
Churn Model Training Script
Run: python training/train.py
Requires: data/telco.csv in project root
"""

import os, sys, json
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    confusion_matrix, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "telco.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "backend", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. Load & Clean ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading and cleaning data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"  Raw shape: {df.shape}")

# Fix TotalCharges — spaces for brand-new customers
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")

# Smart imputation: new customers have TotalCharges ≈ MonthlyCharges × tenure
mask = df["TotalCharges"].isna()
df.loc[mask, "TotalCharges"] = df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
print(f"  Fixed {mask.sum()} TotalCharges NaN rows via imputation")

df = df.drop(columns=["customerID"], errors="ignore")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(f"  Churn distribution: {df['Churn'].value_counts().to_dict()}")

# ─── 2. Encode ───────────────────────────────────────────────────────────────
print("\nSTEP 2: Encoding features")

BINARY_MAP = {"Yes": 1, "No": 0}
MULTI_MAP  = {"No": 0, "Yes": 1, "No internet service": 2, "No phone service": 2}

df["gender"]          = df["gender"].map({"Male": 1, "Female": 0})
df["Partner"]         = df["Partner"].map(BINARY_MAP)
df["Dependents"]      = df["Dependents"].map(BINARY_MAP)
df["PhoneService"]    = df["PhoneService"].map(BINARY_MAP)
df["PaperlessBilling"]= df["PaperlessBilling"].map(BINARY_MAP)

service_cols = ["MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
for col in service_cols:
    df[col] = df[col].map(MULTI_MAP)

df["InternetService"] = df["InternetService"].map({"DSL": 0, "Fiber optic": 1, "No": 2})
df["Contract"]        = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})
df["PaymentMethod"]   = df["PaymentMethod"].map({
    "Electronic check": 0, "Mailed check": 1,
    "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
})

# ─── 3. Feature Engineering ──────────────────────────────────────────────────
print("STEP 3: Engineering features")

df["ChargesPerMonth"]    = df["TotalCharges"] / (df["tenure"] + 1)
df["ServicesCount"]      = df[service_cols].apply(lambda r: (r == 1).sum(), axis=1)
df["IsNewCustomer"]      = (df["tenure"] < 6).astype(int)
df["IsMidCustomer"]      = ((df["tenure"] >= 6) & (df["tenure"] < 24)).astype(int)
df["HighValueAtRisk"]    = ((df["MonthlyCharges"] > 70) & (df["tenure"] < 12)).astype(int)
df["NoSupportServices"]  = ((df["OnlineSecurity"] == 0) & (df["TechSupport"] == 0)).astype(int)
df["AutoPay"]            = (df["PaymentMethod"] >= 2).astype(int)
df["FiberHighCharges"]   = ((df["InternetService"] == 1) & (df["MonthlyCharges"] > 80)).astype(int)
df["M2MNoSecurity"]      = ((df["Contract"] == 0) & (df["OnlineSecurity"] == 0)).astype(int)
df["TenureChargesRatio"] = df["tenure"] / (df["MonthlyCharges"] + 1)

engineered = ["ChargesPerMonth","ServicesCount","IsNewCustomer","IsMidCustomer",
              "HighValueAtRisk","NoSupportServices","AutoPay","FiberHighCharges",
              "M2MNoSecurity","TenureChargesRatio"]
print(f"  Added {len(engineered)} engineered features")

# ─── 4. Prepare X, y ─────────────────────────────────────────────────────────
feature_columns = [c for c in df.columns if c != "Churn"]
X = df[feature_columns].fillna(0)
y = df["Churn"]

print(f"\n  Feature matrix shape: {X.shape}")
print(f"  Features: {feature_columns}")

# ─── 5. Scale numerics ───────────────────────────────────────────────────────
print("\nSTEP 4: Scaling numeric features")
num_cols = ["tenure","MonthlyCharges","TotalCharges","ChargesPerMonth",
            "ServicesCount","TenureChargesRatio"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = X_train_raw.copy()
X_test  = X_test_raw.copy()
X_train[num_cols] = scaler.fit_transform(X_train_raw[num_cols])
X_test[num_cols]  = scaler.transform(X_test_raw[num_cols])

# ─── 6. SMOTE ────────────────────────────────────────────────────────────────
print("STEP 5: Applying SMOTE for class balance")
sm = SMOTE(random_state=42, sampling_strategy=0.6, k_neighbors=5)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
print(f"  Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  After  SMOTE: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

# ─── 7. XGBoost + RandomizedSearch ──────────────────────────────────────────
print("\nSTEP 6: Hyperparameter tuning with RandomizedSearchCV (30 iterations)")

param_grid = {
    "n_estimators":      [200, 300, 400, 500, 600],
    "max_depth":         [3, 4, 5, 6],
    "learning_rate":     [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample":         [0.7, 0.75, 0.8, 0.85, 0.9],
    "colsample_bytree":  [0.7, 0.75, 0.8, 0.85, 0.9],
    "min_child_weight":  [1, 2, 3, 5],
    "gamma":             [0, 0.1, 0.2, 0.3],
    "reg_alpha":         [0, 0.01, 0.1, 0.5],
    "reg_lambda":        [1, 1.5, 2],
    "scale_pos_weight":  [1, 2, 3],
}

base_model = xgb.XGBClassifier(
    eval_metric="auc",
    random_state=42,
    use_label_encoder=False,
    tree_method="hist",
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    base_model, param_grid,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
search.fit(X_resampled, y_resampled)

model = search.best_estimator_
print(f"\n  Best params: {search.best_params_}")
print(f"  CV AUC:      {search.best_score_:.4f}")

# ─── 8. Threshold Calibration ─────────────────────────────────────────────────
print("\nSTEP 7: Calibrating decision threshold")
y_probs = model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.25, 0.65, 0.01)
f1_scores  = [f1_score(y_test, (y_probs >= t).astype(int)) for t in thresholds]
best_threshold = float(thresholds[np.argmax(f1_scores)])
print(f"  Optimal threshold: {best_threshold:.2f} (F1={max(f1_scores):.4f})")

# ─── 9. Final Evaluation ─────────────────────────────────────────────────────
print("\nSTEP 8: Final evaluation on held-out test set")
y_pred_calibrated = (y_probs >= best_threshold).astype(int)

auc = roc_auc_score(y_test, y_probs)
f1  = f1_score(y_test, y_pred_calibrated)
cm  = confusion_matrix(y_test, y_pred_calibrated)

print(f"\n  AUC-ROC:  {auc:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"\n  Confusion Matrix:\n{cm}")
print(f"\n  Classification Report:\n{classification_report(y_test, y_pred_calibrated, target_names=['No Churn','Churn'])}")

# ─── 10. Save Artifacts ──────────────────────────────────────────────────────
print("STEP 9: Saving model artifacts")

joblib.dump(model,            os.path.join(MODEL_DIR, "churn_model.pkl"))
joblib.dump(scaler,           os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(feature_columns,  os.path.join(MODEL_DIR, "feature_columns.pkl"))
joblib.dump(best_threshold,   os.path.join(MODEL_DIR, "threshold.pkl"))

meta = {
    "model_type":       "XGBoostClassifier",
    "version":          "2.0.0",
    "features":         len(feature_columns),
    "feature_names":    feature_columns,
    "training_auc":     round(search.best_score_, 4),
    "test_auc":         round(auc, 4),
    "test_f1":          round(f1, 4),
    "threshold":        round(best_threshold, 3),
    "dataset_size":     len(df),
    "churn_base_rate":  round(float(y.mean()), 4),
    "best_params":      search.best_params_,
    "num_cols_scaled":  num_cols,
}
with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n  Saved to {MODEL_DIR}/")
print("  - churn_model.pkl")
print("  - scaler.pkl")
print("  - feature_columns.pkl")
print("  - threshold.pkl")
print("  - model_meta.json")
print("\n✅ Training complete!")
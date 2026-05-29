"""
Data Service — loads telco.csv and transforms it into API-ready dicts.
"""

import os
import pandas as pd
from functools import lru_cache
from services.ml_service import get_artifacts, preprocess_for_inference

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "telco.csv")


def _compute_risk_score(row: pd.Series) -> int:
    """Heuristic risk score (0–100) for display in the customers table."""
    risk = 20
    if str(row.get("Churn", "")).lower() == "yes":
        risk = 90
    else:
        tenure = int(row.get("tenure", 72))
        if tenure < 6:
            risk += 35
        elif tenure < 12:
            risk += 20
        elif tenure < 24:
            risk += 10

        if float(row.get("MonthlyCharges", 0)) > 70:
            risk += 10
        if str(row.get("Contract", "")).lower() == "month-to-month":
            risk += 8
        if str(row.get("InternetService", "")).lower() == "fiber optic":
            risk += 5
        if str(row.get("PaymentMethod", "")).lower() == "electronic check":
            risk += 7
        if str(row.get("OnlineSecurity", "")).lower() == "no":
            risk += 5

    return min(100, risk)


def load_raw_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = (
        df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
    )
    return df


def get_customers_list() -> list[dict]:
    df = load_raw_df()
    customers = []

    for _, row in df.iterrows():
        customers.append({
            # Identity
            "customerID":       row["customerID"],
            # Demographics
            "gender":           row["gender"],
            "seniorCitizen":    int(row["SeniorCitizen"]),
            "partner":          row["Partner"],
            "dependents":       row["Dependents"],
            # Account
            "tenure":           int(row["tenure"]),
            # Services
            "phoneService":     row["PhoneService"],
            "multipleLines":    row["MultipleLines"],
            "internetService":  row["InternetService"],
            "onlineSecurity":   row["OnlineSecurity"],
            "onlineBackup":     row["OnlineBackup"],
            "deviceProtection": row["DeviceProtection"],
            "techSupport":      row["TechSupport"],
            "streamingTV":      row["StreamingTV"],
            "streamingMovies":  row["StreamingMovies"],
            # Billing
            "contract":         row["Contract"],
            "paperlessBilling": row["PaperlessBilling"],
            "paymentMethod":    row["PaymentMethod"],
            "monthlyCharges":   float(row["MonthlyCharges"]),
            "totalCharges":     float(row["TotalCharges"]),
            # Computed
            "riskScore":        _compute_risk_score(row),
            "churned":          str(row["Churn"]).lower() == "yes",
        })

    return customers


def get_overview_stats() -> dict:
    df = load_raw_df()
    churn_mask = df["Churn"].str.lower() == "yes"
    return {
        "total_customers":   len(df),
        "churn_rate":        round(100 * churn_mask.mean(), 2),
        "avg_revenue":       round(float(df["MonthlyCharges"].mean()), 2),
        "at_risk_customers": int(churn_mask.sum()),
    }


def get_segments() -> list[dict]:
    df = load_raw_df()
    segments = []
    for contract, group in df.groupby("Contract"):
        rate = (group["Churn"].str.lower() == "yes").mean()
        segments.append({
            "name":        contract,
            "customers":   len(group),
            "churn_rate":  round(100 * rate, 2),
            "avg_revenue": round(float(group["MonthlyCharges"].mean()), 2),
            "risk_level":  "High" if rate > 0.4 else "Medium" if rate > 0.15 else "Low",
        })
    return segments


def get_churn_trend() -> dict:
    df = load_raw_df()
    bins   = [0, 6, 12, 24, 36, 48, 60, 72]
    labels = ["0–6", "7–12", "13–24", "25–36", "37–48", "49–60", "61–72"]
    df["tenure_bin"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    trend = (
        df.groupby("tenure_bin", observed=True)["Churn"]
        .apply(lambda x: round(100 * (x.str.lower() == "yes").mean(), 2))
        .reset_index()
    )
    return {
        "labels":      trend["tenure_bin"].astype(str).tolist(),
        "churn_rates": trend["Churn"].tolist(),
    }
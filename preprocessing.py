"""
Preprocessing pipeline — shared between training and inference.
Both paths must apply IDENTICAL transformations.
"""

import pandas as pd
import numpy as np

BINARY_MAP = {"Yes": 1, "No": 0}
MULTI_MAP  = {"No": 0, "Yes": 1, "No internet service": 2, "No phone service": 2}

SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

NUM_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "ChargesPerMonth", "ServicesCount", "TenureChargesRatio",
]


def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Encode a raw telco CSV dataframe (training path)."""
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = (
        df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
    )

    df["gender"]           = df["gender"].map({"Male": 1, "Female": 0})
    df["Partner"]          = df["Partner"].map(BINARY_MAP)
    df["Dependents"]       = df["Dependents"].map(BINARY_MAP)
    df["PhoneService"]     = df["PhoneService"].map(BINARY_MAP)
    df["PaperlessBilling"] = df["PaperlessBilling"].map(BINARY_MAP)

    for col in SERVICE_COLS:
        df[col] = df[col].map(MULTI_MAP)

    df["InternetService"] = df["InternetService"].map(
        {"DSL": 0, "Fiber optic": 1, "No": 2}
    )
    df["Contract"] = df["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    )
    df["PaymentMethod"] = df["PaymentMethod"].map({
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3,
    })

    df = _add_engineered_features(df)
    return df


def encode_api_input(data: dict) -> pd.DataFrame:
    """
    Encode a single prediction request dict (inference path).
    Values are already numeric (0/1/2/3) from the frontend.
    Only engineered features need to be derived.
    """
    df = pd.DataFrame([data])
    df = _add_engineered_features(df)
    return df


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ChargesPerMonth"]   = df["TotalCharges"] / (df["tenure"] + 1)
    df["ServicesCount"]     = df[SERVICE_COLS].apply(lambda r: (r == 1).sum(), axis=1)
    df["IsNewCustomer"]     = (df["tenure"] < 6).astype(int)
    df["IsMidCustomer"]     = ((df["tenure"] >= 6) & (df["tenure"] < 24)).astype(int)
    df["HighValueAtRisk"]   = (
        (df["MonthlyCharges"] > 70) & (df["tenure"] < 12)
    ).astype(int)
    df["NoSupportServices"] = (
        (df["OnlineSecurity"] == 0) & (df["TechSupport"] == 0)
    ).astype(int)
    df["AutoPay"]           = (df["PaymentMethod"] >= 2).astype(int)
    df["FiberHighCharges"]  = (
        (df["InternetService"] == 1) & (df["MonthlyCharges"] > 80)
    ).astype(int)
    df["M2MNoSecurity"]     = (
        (df["Contract"] == 0) & (df["OnlineSecurity"] == 0)
    ).astype(int)
    df["TenureChargesRatio"]= df["tenure"] / (df["MonthlyCharges"] + 1)
    return df
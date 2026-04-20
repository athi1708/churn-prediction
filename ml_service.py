"""
ML Service — loads model artifacts once and exposes prediction helpers.
Uses functools.lru_cache to guarantee a single load per process.
"""

import os, json
import joblib
import pandas as pd
import numpy as np
from functools import lru_cache

from utils.preprocessing import encode_api_input, NUM_COLS
from utils.explain import explain_prediction
from utils.insights import generate_insights, get_primary_action
from utils.confidence import interpret_confidence

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "model")


@lru_cache(maxsize=1)
def _load_artifacts() -> dict:
    """Load once, cache forever — no repeated disk I/O per request."""
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return {
        "model":     joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl")),
        "scaler":    joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        "columns":   joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl")),
        "threshold": joblib.load(os.path.join(MODEL_DIR, "threshold.pkl")),
        "meta":      meta,
    }


def get_artifacts() -> dict:
    return _load_artifacts()


def preprocess_for_inference(data: dict) -> pd.DataFrame:
    arts = get_artifacts()
    df   = encode_api_input(data)

    # Scale numeric columns with the SAME scaler used in training
    cols_present = [c for c in NUM_COLS if c in df.columns]
    df[cols_present] = arts["scaler"].transform(df[cols_present])

    # Align to training feature order, fill any missing with 0
    for col in arts["columns"]:
        if col not in df.columns:
            df[col] = 0
    return df[arts["columns"]].fillna(0)


def run_prediction(data: dict) -> dict:
    arts      = get_artifacts()
    model     = arts["model"]
    threshold = arts["threshold"]

    df   = preprocess_for_inference(data)
    prob = float(model.predict_proba(df)[0][1])
    pred = int(prob >= threshold)

    top_reasons = explain_prediction(df, top_n=6)
    insights    = generate_insights(data, top_reasons)
    action      = get_primary_action(data, pred, prob)
    confidence  = interpret_confidence(prob, top_reasons, pred)

    return {
    "churn": int(pred),
    "probability": float(round(prob, 4)),
    "top_reasons": [[f, round(float(v), 4)] for f, v in top_reasons],
    "insights": insights,
    "action": action,
    "confidence": {
        "label": confidence.label,
        "score": float(round(confidence.score, 3)),
        "message": confidence.message,
        "recommendation": confidence.recommendation,
        "color": confidence.color,
        "icon": confidence.icon,
    },
}
"""
Monitoring — logs every prediction and exposes drift/stats endpoint.
In production replace the in-memory list with Redis or a database.
"""

from fastapi import APIRouter
from datetime import datetime, timezone
from collections import deque
import statistics

router = APIRouter(tags=["Monitoring"])

# In-memory ring buffer — keeps last 10,000 predictions
_log: deque = deque(maxlen=10_000)


def log_prediction(data: dict, result: dict):
    """Called by the predict router after each successful prediction."""
    _log.append({
        "ts":          datetime.now(timezone.utc).isoformat(),
        "probability": result["probability"],
        "churn":       result["churn"],
        "contract":    data.get("Contract"),
        "tenure":      data.get("tenure"),
        "monthly":     data.get("MonthlyCharges"),
        "confidence":  result["confidence"]["label"],
    })


@router.get("/monitoring/stats")
def monitoring_stats():
    if not _log:
        return {"message": "No predictions logged yet.", "total": 0}

    probs       = [p["probability"] for p in _log]
    churn_flags = [p["churn"] for p in _log]

    avg_prob   = statistics.mean(probs)
    std_prob   = statistics.stdev(probs) if len(probs) > 1 else 0.0
    drift_flag = std_prob > 0.25  # simple variance-based drift signal

    high_conf  = sum(1 for p in _log if p["confidence"] == "High Confidence")
    low_conf   = sum(1 for p in _log if p["confidence"] == "Low Confidence")

    return {
        "total_predictions":    len(_log),
        "churn_predictions":    sum(churn_flags),
        "non_churn_predictions":len(churn_flags) - sum(churn_flags),
        "avg_churn_probability":round(avg_prob, 4),
        "std_churn_probability":round(std_prob, 4),
        "high_risk_count":      sum(1 for p in probs if p > 0.7),
        "high_confidence_pct":  round(100 * high_conf / len(_log), 1),
        "low_confidence_pct":   round(100 * low_conf  / len(_log), 1),
        "data_drift_alert":     drift_flag,
        "drift_reason":         "High probability variance detected — check for distribution shift." if drift_flag else None,
    }


@router.delete("/monitoring/reset")
def reset_log():
    _log.clear()
    return {"message": "Prediction log cleared."}
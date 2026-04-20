"""
/predict  and  /batch_predict  routes
"""

from fastapi import APIRouter, HTTPException
from models.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse, BatchPredictItem,
)
from services.ml_service import run_prediction

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    try:
        result = run_prediction(data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(payload: BatchPredictRequest):
    """Score up to 200 customers in a single request."""
    results = []
    high_risk = 0

    for i, customer in enumerate(payload.customers):
        try:
            r = run_prediction(customer.model_dump())
            if r["churn"] == 1:
                high_risk += 1
            results.append(BatchPredictItem(
                index=i,
                churn=r["churn"],
                probability=r["probability"],
                confidence=r["confidence"]["label"],
            ))
        except Exception as e:
            results.append(BatchPredictItem(
                index=i, churn=-1, probability=-1.0, confidence=f"Error: {e}"
            ))

    return BatchPredictResponse(
        total=len(results),
        high_risk=high_risk,
        results=results,
    )
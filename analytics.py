from fastapi import APIRouter
from services.data_service import get_overview_stats, get_segments, get_churn_trend

router = APIRouter(tags=["Analytics"])


@router.get("/overview")
def overview():
    return get_overview_stats()


@router.get("/segments")
def segments():
    return get_segments()


@router.get("/churn_trend")
def churn_trend():
    return get_churn_trend()
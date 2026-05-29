"""
Churn Intelligence API — main entry point
Run: uvicorn main:app --reload --port 8000
"""

import os, sys, json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Make backend/ importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from routers import predict, customers, analytics, monitoring
from routers.monitoring import log_prediction
from services.ml_service import get_artifacts

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Intelligence API",
    description="Production-grade customer churn prediction for telco data.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global exception handler ────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(customers.router)
app.include_router(analytics.router)
app.include_router(monitoring.router)

# ─── Middleware: log every prediction ────────────────────────────────────────
# We patch the predict endpoint response after it's returned so the router
# stays clean. Alternative: use a background task inside the router.
@app.middleware("http")
async def prediction_logger(request: Request, call_next):
    response = await call_next(request)
    # Logging is handled directly inside predict router via log_prediction()
    return response

# ─── Root ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Churn Intelligence API v2.0 🚀", "docs": "/docs"}


@app.get("/model/info")
def model_info():
    """Returns metadata about the currently loaded model."""
    arts = get_artifacts()
    meta = arts.get("meta", {})
    return {
        "model_type":      meta.get("model_type", "XGBoostClassifier"),
        "version":         meta.get("version", "2.0.0"),
        "features":        meta.get("features", len(arts["columns"])),
        "feature_names":   arts["columns"],
        "training_auc":    meta.get("training_auc", "N/A"),
        "test_auc":        meta.get("test_auc", "N/A"),
        "test_f1":         meta.get("test_f1", "N/A"),
        "threshold":       float(arts["threshold"]),
        "dataset_size":    meta.get("dataset_size", "N/A"),
        "churn_base_rate": meta.get("churn_base_rate", "N/A"),
    }


# ─── Startup ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    """Pre-warm the model cache on startup to avoid cold-start latency."""
    print("🔄 Pre-loading ML model artifacts...")
    try:
        arts = get_artifacts()
        print(f"✅ Model loaded: {arts['meta'].get('model_type', 'XGBoost')}")
        print(f"   Threshold: {arts['threshold']:.3f}")
        print(f"   Features:  {len(arts['columns'])}")
    except Exception as e:
        print(f"⚠️  Model not found: {e}")
        print("   Run  python training/train.py  first.")
"""
Pydantic schemas — enforce input types/ranges before any ML processing.
This catches bad data at the API boundary, not inside the model.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional


class PredictRequest(BaseModel):
    tenure:           int   = Field(..., ge=0,   le=72,  description="Months as customer (0–72)")
    SeniorCitizen:    int   = Field(..., ge=0,   le=1,   description="0=No, 1=Yes")
    Partner:          int   = Field(..., ge=0,   le=1,   description="0=No, 1=Yes")
    Dependents:       int   = Field(..., ge=0,   le=1,   description="0=No, 1=Yes")
    PhoneService:     int   = Field(..., ge=0,   le=1,   description="0=No, 1=Yes")
    MultipleLines:    int   = Field(..., ge=0,   le=2,   description="0=No, 1=Yes, 2=No phone svc")
    InternetService:  int   = Field(..., ge=0,   le=2,   description="0=DSL, 1=Fiber optic, 2=No")
    OnlineSecurity:   int   = Field(..., ge=0,   le=2,   description="0=No, 1=Yes, 2=No internet")
    OnlineBackup:     int   = Field(..., ge=0,   le=2)
    DeviceProtection: int   = Field(..., ge=0,   le=2)
    TechSupport:      int   = Field(..., ge=0,   le=2)
    StreamingTV:      int   = Field(..., ge=0,   le=2)
    StreamingMovies:  int   = Field(..., ge=0,   le=2)
    Contract:         int   = Field(..., ge=0,   le=2,   description="0=M2M, 1=1yr, 2=2yr")
    PaperlessBilling: int   = Field(..., ge=0,   le=1)
    PaymentMethod:    int   = Field(..., ge=0,   le=3,   description="0=eCheck, 1=Mail, 2=BankAuto, 3=CardAuto")
    MonthlyCharges:   float = Field(..., ge=0.0, le=500.0)
    TotalCharges:     float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def validate_charges_consistency(self):
        if self.TotalCharges > 0 and self.TotalCharges < self.MonthlyCharges * 0.5:
            raise ValueError(
                f"TotalCharges ({self.TotalCharges}) looks too low for "
                f"MonthlyCharges ({self.MonthlyCharges}) and tenure ({self.tenure}). "
                "Expected TotalCharges ≈ tenure × MonthlyCharges."
            )
        return self


class ConfidenceOut(BaseModel):
    label:          str
    score:          float
    message:        str
    recommendation: str
    color:          str
    icon:           str


class PredictResponse(BaseModel):
    churn:       int
    probability: float
    top_reasons: list
    insights:    list[str]
    action:      str
    confidence:  ConfidenceOut


class BatchPredictRequest(BaseModel):
    customers: list[PredictRequest] = Field(..., max_length=200)


class BatchPredictItem(BaseModel):
    index:       int
    churn:       int
    probability: float
    confidence:  str


class BatchPredictResponse(BaseModel):
    total:          int
    high_risk:      int
    results:        list[BatchPredictItem]
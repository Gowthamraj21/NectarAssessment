"""FastAPI serving layer for the trained predictive-maintenance model."""
from pathlib import Path
from typing import Optional, List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
import sys
sys.path.insert(0, str(ROOT / "src"))
from predictive_features import engineer_predictive_features
model = joblib.load(MODEL_DIR / "predictive_maintenance_model.joblib")
feature_cols = joblib.load(MODEL_DIR / "predictive_maintenance_features.joblib")
preprocess = joblib.load(MODEL_DIR / "predictive_maintenance_preprocessing.joblib") if (MODEL_DIR / "predictive_maintenance_preprocessing.joblib").exists() else {"threshold": .5, "medians": {}}
THRESHOLD = float(preprocess.get("threshold", .5))
MEDIANS = preprocess.get("medians", {})

app = FastAPI(title="Nectar Predictive Maintenance API", description="Predicts probability of asset failure within the next 24 clock hours.", version="2.0.0")

class TelemetryReading(BaseModel):
    timestamp: str = Field(..., example="2026-07-20T14:00:00")
    asset_type: str = Field(..., example="Chiller")
    operating_mode: str = Field(..., example="Cooling")
    temperature: float
    humidity: float
    pressure: float
    vibration: float
    power_consumption: float
    occupancy_count: int
    capacity: Optional[float] = None

class PredictionRequest(BaseModel):
    current: TelemetryReading
    history: Optional[List[TelemetryReading]] = None

class PredictionResponse(BaseModel):
    failure_probability_24h: float
    risk_tier: str
    decision_threshold: float
    model_used: str
    history_points_used: int

def _build_feature_row(req: PredictionRequest) -> pd.DataFrame:
    records = [h.model_dump() for h in (req.history or [])] + [req.current.model_dump()]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # The endpoint receives one asset's history, so shared training features apply directly.
    df["asset_id"] = "request_asset"
    df = engineer_predictive_features(df, medians=MEDIANS)
    row = df.iloc[[-1]].copy()
    for c in feature_cols:
        if c not in row.columns: row[c] = 0
    return row[feature_cols]

@app.get("/")
def root(): return {"status":"ok","service":"Nectar Predictive Maintenance API","model":type(model).__name__}

@app.post("/predict_failure", response_model=PredictionResponse)
def predict_failure(req: PredictionRequest):
    try: X=_build_feature_row(req)
    except Exception as e: raise HTTPException(status_code=422, detail=f"Feature construction failed: {e}")
    proba=float(model.predict_proba(X)[:,1][0])
    tier="HIGH" if proba>=THRESHOLD else "MEDIUM" if proba>=THRESHOLD/2 else "LOW"
    return PredictionResponse(failure_probability_24h=round(proba,4),risk_tier=tier,decision_threshold=round(THRESHOLD,4),model_used=type(model).__name__,history_points_used=len(req.history or []))

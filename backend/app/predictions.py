from functools import lru_cache
from pathlib import Path
import math
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from app.auth import current_user

router = APIRouter(tags=["Risk Predictions"])
ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "outputs" / "phase_h" / "slidronix_phase_h_risk_predictions.csv"

class CoordinateRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

@lru_cache(maxsize=1)
def get_predictions():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Phase H predictions not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    required = {"Latitude", "Longitude", "Landslide_Probability", "Risk_Level"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df = df.dropna(subset=list(required)).copy()
    for col in ("Latitude", "Longitude", "Landslide_Probability"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude", "Landslide_Probability"])
    if df.empty:
        raise ValueError("No valid prediction rows")
    if not df["Latitude"].between(-90, 90).all() or not df["Longitude"].between(-180, 180).all():
        raise ValueError("Invalid coordinate values in dataset")
    if not df["Landslide_Probability"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1")
    return df.reset_index(drop=True)

def data_or_503():
    try:
        return get_predictions()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2-lat1, lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(a)))

@router.get("/predictions/summary")
def prediction_summary(user=Depends(current_user)):
    df = data_or_503()
    counts = df["Risk_Level"].astype(str).value_counts()
    return {
        "source": "precomputed_phase_h_model_predictions",
        "total_locations": int(len(df)),
        "risk_counts": {level: int(counts.get(level, 0)) for level in ("Low", "Moderate", "High")},
        "mean_probability": round(float(df["Landslide_Probability"].mean()), 6),
        "thresholds": {"low": "< 0.33", "moderate": "0.33 to < 0.66", "high": ">= 0.66"},
        "warning": "Research prototype using historical data; not a validated operational warning."
    }

@router.get("/predictions/points")
def prediction_points(risk_level: str | None = Query(default=None), limit: int = Query(default=100, ge=1, le=1000), offset: int = Query(default=0, ge=0), user=Depends(current_user)):
    df = data_or_503()
    if risk_level is not None:
        if risk_level not in {"Low", "Moderate", "High"}:
            raise HTTPException(status_code=422, detail="risk_level must be Low, Moderate, or High")
        df = df[df["Risk_Level"].astype(str) == risk_level]
    total = len(df)
    page = df.iloc[offset:offset+limit]
    return {
        "source": "precomputed_phase_h_model_predictions",
        "total_matching": int(total), "offset": offset, "limit": limit,
        "points": [{
            "latitude": float(row["Latitude"]), "longitude": float(row["Longitude"]),
            "landslide_probability": float(row["Landslide_Probability"]), "risk_level": str(row["Risk_Level"])
        } for _, row in page.iterrows()]
    }

@router.post("/predict/nearest")
def predict_nearest(request: CoordinateRequest, user=Depends(current_user)):
    df = data_or_503()
    distances = [haversine_km(request.latitude, request.longitude, float(row["Latitude"]), float(row["Longitude"])) for _, row in df.iterrows()]
    index = min(range(len(distances)), key=distances.__getitem__)
    row = df.iloc[index]
    return {
        "requested_location": {"latitude": request.latitude, "longitude": request.longitude},
        "matched_location": {"latitude": float(row["Latitude"]), "longitude": float(row["Longitude"])},
        "distance_km": round(distances[index], 3),
        "landslide_probability": float(row["Landslide_Probability"]),
        "risk_level": str(row["Risk_Level"]),
        "source": "nearest_precomputed_phase_h_prediction",
        "is_new_model_inference": False,
        "warning": "Nearest historical prediction only; not a fresh prediction or live warning."
    }

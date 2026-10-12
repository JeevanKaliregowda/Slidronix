from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from app.auth import current_user
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

router = APIRouter(tags=["live-inference"])
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
TRAIN_FILE = DATA / "phase_e" / "slidronix_phase_e_train.csv"
CNN_FILE = DATA / "phase_f" / "cnn_spatial" / "train_cnn_patches.npz"
RAIN_FILE = DATA / "rainfall" / "chirps_2022_daily_14270.csv"
MODEL_FILE = ROOT / "outputs" / "phase_f" / "fusion_best.pt"
DEM_DIR = ROOT / "data" / "raw" / "dem"

FEATURES = [
    "Elevation_m", "Slope_deg",
    "rainfall_total_30d_mm", "rainfall_mean_daily_mm",
    "rainfall_max_daily_mm", "rainfall_std_daily_mm",
    "rainfall_7d_mm", "rainfall_15d_mm", "rainfall_30d_mm",
    "rainfall_max_7d_mm", "rainfall_max_15d_mm", "rainfall_max_30d_mm",
    "NDVI_mean", "NDVI_std", "NDVI_min", "NDVI_max",
    "NDVI_valid_observations", "NDVI_range",
    "Land_Cover_Type", "Land_Cover_QC",
    "Soil_clay_0_5cm", "Soil_sand_0_5cm", "Soil_silt_0_5cm",
    "Soil_soc_0_5cm", "Soil_bdod_0_5cm", "Soil_phh2o_0_5cm",
]
STATIC_FEATURES = [
    "NDVI_mean", "NDVI_std", "NDVI_min", "NDVI_max",
    "NDVI_valid_observations", "NDVI_range", "Land_Cover_Type",
    "Land_Cover_QC", "Soil_clay_0_5cm", "Soil_sand_0_5cm",
    "Soil_silt_0_5cm", "Soil_soc_0_5cm", "Soil_bdod_0_5cm",
    "Soil_phh2o_0_5cm",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionModel(nn.Module):
    def __init__(self, tabular_features: int):
        super().__init__()
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_features, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, 32)
        )
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.cnn_projection = nn.Sequential(nn.Linear(128, 32), nn.ReLU())
        self.rain_input = nn.Linear(1, 64)
        self.rain_position = nn.Parameter(torch.zeros(1, 30, 64))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.2,
            batch_first=True, activation="gelu"
        )
        self.rain_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.rain_projection = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(96, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, tabular, cnn, rainfall):
        tab_embedding = self.tabular_encoder(tabular)
        cnn_features = self.cnn_encoder(cnn).flatten(1)
        cnn_embedding = self.cnn_projection(cnn_features)
        rainfall = self.rain_input(rainfall) + self.rain_position
        rainfall = self.rain_transformer(rainfall).mean(dim=1)
        rain_embedding = self.rain_projection(rainfall)
        fused = torch.cat([tab_embedding, cnn_embedding, rain_embedding], dim=1)
        return self.fusion(fused).squeeze(1)


def norm_id(value: Any) -> str:
    return str(value).replace("_", "").strip()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def find_dem(lat: float, lon: float):
    if DEM_DIR.exists():
        for path in DEM_DIR.glob("*.tif"):
            try:
                with rasterio.open(path) as src:
                    b = src.bounds
                    if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
                        return path
            except Exception:
                continue
    # Copernicus GLO-30 public COG tile convention used by the project acquisition script.
    lat_deg, lon_deg = math.floor(lat), math.floor(lon)
    lat_tag = f"N{lat_deg:02d}" if lat_deg >= 0 else f"S{-lat_deg:02d}"
    lon_tag = f"E{lon_deg:03d}" if lon_deg >= 0 else f"W{-lon_deg:03d}"
    tile = f"Copernicus_DSM_COG_10_{lat_tag}_00_{lon_tag}_00_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif"


def extract_dem_patch(lat: float, lon: float):
    source = find_dem(lat, lon)
    try:
        with rasterio.open(source) as src:
            b = src.bounds
            if not (b.left <= lon <= b.right and b.bottom <= lat <= b.top):
                raise ValueError("The available DEM tile does not cover this coordinate.")
            row, col = src.index(lon, lat)
            half = 16
            window = Window(col - half, row - half, 32, 32)
            elevation = src.read(1, window=window, boundless=True, fill_value=np.nan).astype(np.float32)
            if elevation.shape != (32, 32):
                raise ValueError("Could not construct a 32x32 terrain patch.")
            if src.nodata is not None:
                elevation[elevation == src.nodata] = np.nan
            if np.isfinite(elevation).mean() < 0.95:
                raise ValueError("Terrain patch contains too many missing DEM cells.")
            elevation = np.where(np.isfinite(elevation), elevation, np.nanmedian(elevation))
            transform = src.window_transform(window)
            lat_center = lat
            px_w = abs(transform.a) * 111320.0 * math.cos(math.radians(lat_center))
            px_h = abs(transform.e) * 111320.0
            gy, gx = np.gradient(elevation, px_h, px_w)
            slope = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
            slope = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            return np.stack([elevation, slope]).astype(np.float32), float(elevation[16, 16]), float(slope[16, 16]), str(source)
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain a valid 32x32 Copernicus DEM terrain patch for this coordinate. "
            "Check that the coordinate is within the Western Ghats study coverage and that the DEM tile is available. "
            f"Details: {exc}"
        ) from exc


def fetch_live_rainfall(lat: float, lon: float) -> np.ndarray:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "precipitation_sum",
        "past_days": 31, "forecast_days": 1, "timezone": "UTC",
    }
    try:
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        payload = response.json()
        values = payload.get("daily", {}).get("precipitation_sum", [])
        values = [float(v) if v is not None else np.nan for v in values]
        values = np.asarray(values, dtype=np.float32)
        values = values[np.isfinite(values)]
        if len(values) < 30:
            raise ValueError(f"API returned only {len(values)} usable daily rainfall values.")
        return values[-30:]
    except Exception as exc:
        raise RuntimeError(f"Live rainfall API request failed: {exc}") from exc


def rainfall_features(values: np.ndarray) -> dict[str, float]:
    last7, last15 = values[-7:], values[-15:]
    return {
        "rainfall_total_30d_mm": float(np.sum(values)),
        "rainfall_mean_daily_mm": float(np.mean(values)),
        "rainfall_max_daily_mm": float(np.max(values)),
        "rainfall_std_daily_mm": float(np.std(values, ddof=1)),
        "rainfall_7d_mm": float(np.sum(last7)),
        "rainfall_15d_mm": float(np.sum(last15)),
        "rainfall_30d_mm": float(np.sum(values)),
        "rainfall_max_7d_mm": float(np.max(last7)),
        "rainfall_max_15d_mm": float(np.max(last15)),
        "rainfall_max_30d_mm": float(np.max(values)),
    }


@lru_cache(maxsize=1)
def load_context():
    required = [TRAIN_FILE, CNN_FILE, RAIN_FILE, MODEL_FILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Required local model/data files are missing: " + ", ".join(missing))

    train = pd.read_csv(TRAIN_FILE)
    train["ID_norm"] = train["ID"].map(norm_id)
    cnn = np.load(CNN_FILE, allow_pickle=True)
    cnn_x = cnn["X"].astype(np.float32)
    cnn_ids = [norm_id(x) for x in cnn["ids"]]
    cnn_lookup = {sample_id: i for i, sample_id in enumerate(cnn_ids)}
    rain = pd.read_csv(RAIN_FILE)
    rain["ID_norm"] = rain["ID"].map(norm_id)
    daily_cols = [c for c in rain.columns if c not in {"ID", "Latitude", "Longitude", "ID_norm"}][:30]
    if len(daily_cols) != 30:
        raise RuntimeError(f"Expected 30 training rainfall columns; found {len(daily_cols)}.")
    rain_lookup = rain.set_index("ID_norm")
    valid = train[
        train["ID_norm"].isin(cnn_lookup) &
        train["ID_norm"].isin(rain_lookup.index)
    ].copy()
    if valid.empty:
        raise RuntimeError("Could not align training, terrain-patch and rainfall IDs for preprocessing.")
    cnn_indices = [cnn_lookup[x] for x in valid["ID_norm"]]
    train_cnn_aligned = cnn_x[cnn_indices]
    train_rain_aligned = rain_lookup.loc[valid["ID_norm"], daily_cols].to_numpy(dtype=np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(valid[FEATURES])
    scaler.fit(imputed)
    cnn_mean = train_cnn_aligned.mean(axis=(0, 2, 3), keepdims=True)
    cnn_std = train_cnn_aligned.std(axis=(0, 2, 3), keepdims=True)
    cnn_std[cnn_std < 1e-8] = 1.0
    rain_mean = train_rain_aligned.mean(axis=0)
    rain_std = train_rain_aligned.std(axis=0)
    rain_std[rain_std < 1e-8] = 1.0
    model = FusionModel(len(FEATURES)).to(DEVICE)
    state = torch.load(MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Static satellite/soil features come from the project's environmental reference table.
    candidates = [
        DATA / "slidronix_integrated_dataset_14270.csv",
        DATA / "landslide_environmental_features_clean.csv",
        TRAIN_FILE,
    ]
    reference = None
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if {"Latitude", "Longitude", *STATIC_FEATURES}.issubset(df.columns):
                    reference = df.dropna(subset=["Latitude", "Longitude"]).copy()
                    break
            except Exception:
                pass
    if reference is None:
        reference = train.dropna(subset=["Latitude", "Longitude"]).copy()
    if not {"Latitude", "Longitude", *STATIC_FEATURES}.issubset(reference.columns):
        raise RuntimeError("No local reference dataset contains the static satellite, land-cover and soil features.")

    return {
        "train": valid, "reference": reference, "imputer": imputer, "scaler": scaler,
        "cnn_mean": cnn_mean, "cnn_std": cnn_std,
        "rain_mean": rain_mean, "rain_std": rain_std,
        "model": model, "bounds": {
            "min_lat": float(reference["Latitude"].min()),
            "max_lat": float(reference["Latitude"].max()),
            "min_lon": float(reference["Longitude"].min()),
            "max_lon": float(reference["Longitude"].max()),
        },
    }


class CoordinateRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


@router.get("/predict/coordinate/status")
def coordinate_status(user=Depends(current_user)):
    try:
        ctx = load_context()
        return {
            "ready": True,
            "model": "trained_phase_f_fusion_best",
            "feature_count": len(FEATURES),
            "terrain_shape": [2, 32, 32],
            "rainfall_steps": 30,
            "study_bounds": ctx["bounds"],
            "rainfall_provider": "Open-Meteo API",
            "terrain_provider": "local Copernicus DEM tile or public Copernicus COG",
            "static_covariates": "nearest environmental reference sample; not live satellite/soil API data",
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/predict/coordinate")
def predict_coordinate(request: CoordinateRequest, user=Depends(current_user)):
    lat, lon = float(request.latitude), float(request.longitude)
    try:
        ctx = load_context()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Fusion model/data could not load: {exc}") from exc

    bounds = ctx["bounds"]
    if not (bounds["min_lat"] <= lat <= bounds["max_lat"] and bounds["min_lon"] <= lon <= bounds["max_lon"]):
        raise HTTPException(
            status_code=422,
            detail="Coordinate is outside the environmental reference dataset coverage for this Western Ghats prototype. "
                   f"Supported bounding box: lat {bounds['min_lat']:.4f} to {bounds['max_lat']:.4f}, "
                   f"lon {bounds['min_lon']:.4f} to {bounds['max_lon']:.4f}.",
        )

    reference = ctx["reference"]
    # Nearest environmental covariate row only; its old label/probability is never used.
    distances = reference.apply(
        lambda row: haversine_km(lat, lon, float(row["Latitude"]), float(row["Longitude"])),
        axis=1
    )
    ref_idx = distances.idxmin()
    ref = reference.loc[ref_idx]
    static_distance = float(distances.loc[ref_idx])

    try:
        terrain, elevation, slope, dem_source = extract_dem_patch(lat, lon)
        rain_sequence = fetch_live_rainfall(lat, lon)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    values = {feature: float(ref[feature]) if pd.notna(ref[feature]) else np.nan for feature in FEATURES}
    values["Elevation_m"] = elevation
    values["Slope_deg"] = slope
    values.update(rainfall_features(rain_sequence))
    tabular_df = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    tabular = ctx["scaler"].transform(ctx["imputer"].transform(tabular_df)).astype(np.float32)
    terrain_norm = (terrain - ctx["cnn_mean"][0]) / ctx["cnn_std"][0]
    rain_norm = (rain_sequence - ctx["rain_mean"]) / ctx["rain_std"]
    tab_t = torch.tensor(tabular, dtype=torch.float32, device=DEVICE)
    cnn_t = torch.tensor(terrain_norm[None, ...], dtype=torch.float32, device=DEVICE)
    rain_t = torch.tensor(rain_norm[None, :, None], dtype=torch.float32, device=DEVICE)
    try:
        with torch.no_grad():
            logit = ctx["model"](tab_t, cnn_t, rain_t)
            probability = float(torch.sigmoid(logit)[0].cpu().item())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fusion model inference failed: {exc}") from exc

    risk = "Low" if probability < 0.33 else ("Moderate" if probability < 0.66 else "High")
    return {
        "source": "fresh_fusion_model_inference",
        "is_new_model_inference": True,
        "latitude": lat,
        "longitude": lon,
        "landslide_probability": probability,
        "risk_level": risk,
        "features": {
            "Elevation_m": elevation, "Slope_deg": slope,
            **{k: values[k] for k in FEATURES if k not in {"Elevation_m", "Slope_deg"}}
        },
        "data_sources": {
            "rainfall": "Open-Meteo API; last 30 available daily precipitation totals",
            "terrain": "Copernicus DEM 32x32 elevation/slope patch",
            "static_covariates": "Nearest local environmental reference row for NDVI, land cover and soil; not fetched live",
            "static_reference_distance_km": round(static_distance, 3),
            "dem_source": dem_source,
        },
        "warning": "Research prototype. Static NDVI, land-cover and soil values are approximated from the nearest environmental reference sample; this is not an operationally validated warning.",
    }

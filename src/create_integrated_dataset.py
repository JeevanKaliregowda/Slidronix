import pandas as pd
from pathlib import Path

BASE = Path("data/processed")

ENV = pd.read_csv(
    BASE / "landslide_environmental_features_clean.csv"
)

RAIN = pd.read_csv(
    BASE / "rainfall" / "chirps_2022_30day_features_14270.csv"
)

SAT = pd.read_csv(
    BASE / "satellite_features_14270.csv"
)

SOIL = pd.read_csv(
    BASE / "soil_features_14270.csv"
)

POINT_FILES = sorted(
    Path("data/raw/appeears/requests").glob(
        "slidronix_2022_points_*.csv"
    )
)

POINTS = pd.concat(
    [pd.read_csv(f) for f in POINT_FILES],
    ignore_index=True
)

print("=" * 70)
print("SLIDRONIX - FINAL ENVIRONMENTAL DATA INTEGRATION")
print("=" * 70)

print("\nInput sizes:")
print("Environmental:", ENV.shape)
print("Rainfall:", RAIN.shape)
print("Satellite:", SAT.shape)
print("Soil:", SOIL.shape)
print("AppEEARS points:", POINTS.shape)

# =========================================================
# 1. VERIFY MASTER SAMPLE ORDER
# =========================================================

if len(ENV) != len(POINTS):
    raise RuntimeError("Environmental and point datasets have different sizes.")

lat_match = (
    ENV["Latitude"].round(6).to_numpy()
    == POINTS["Latitude"].round(6).to_numpy()
)

lon_match = (
    ENV["Longitude"].round(6).to_numpy()
    == POINTS["Longitude"].round(6).to_numpy()
)

coordinate_matches = (lat_match & lon_match).sum()

print("\nCoordinate matches:", coordinate_matches)
print("Coordinate mismatches:", len(ENV) - coordinate_matches)

if coordinate_matches != len(ENV):
    raise RuntimeError(
        "Master sample order verification failed."
    )

# =========================================================
# 2. ASSIGN VERIFIED APP EEARS ID
# =========================================================

ENV["ID"] = POINTS["ID"].astype(str).str.replace("_", "", regex=False).values

# Soil was extracted from the same environmental dataframe
# order, so assign the same verified IDs by row order.
if len(SOIL) != len(ENV):
    raise RuntimeError("Soil and environmental datasets differ in size.")

soil_lat_match = (
    SOIL["Latitude"].round(6).to_numpy()
    == ENV["Latitude"].round(6).to_numpy()
)

soil_lon_match = (
    SOIL["Longitude"].round(6).to_numpy()
    == ENV["Longitude"].round(6).to_numpy()
)

soil_matches = (soil_lat_match & soil_lon_match).sum()

print("Soil coordinate matches:", soil_matches)

if soil_matches != len(ENV):
    raise RuntimeError(
        "Soil sample order verification failed."
    )

SOIL["ID"] = POINTS["ID"].astype(str).str.replace("_", "", regex=False).values

# =========================================================
# 3. SELECT RAINFALL FEATURES
# =========================================================

rain_cols = [
    "ID",
    "rainfall_total_30d_mm",
    "rainfall_mean_daily_mm",
    "rainfall_max_daily_mm",
    "rainfall_std_daily_mm",
    "rainfall_7d_mm",
    "rainfall_15d_mm",
    "rainfall_30d_mm",
    "rainfall_max_7d_mm",
    "rainfall_max_15d_mm",
    "rainfall_max_30d_mm"
]

RAIN = RAIN[rain_cols].copy()

# =========================================================
# 4. SELECT SATELLITE FEATURES
# =========================================================

sat_cols = [
    "ID",
    "NDVI_mean",
    "NDVI_std",
    "NDVI_min",
    "NDVI_max",
    "NDVI_valid_observations",
    "NDVI_range",
    "Land_Cover_Type",
    "Land_Cover_QC"
]

SAT = SAT[sat_cols].copy()

# =========================================================
# 5. SELECT SOIL FEATURES
# =========================================================

soil_cols = [
    "ID",
    "Soil_clay_0_5cm",
    "Soil_sand_0_5cm",
    "Soil_silt_0_5cm",
    "Soil_soc_0_5cm",
    "Soil_bdod_0_5cm",
    "Soil_phh2o_0_5cm"
]

SOIL = SOIL[soil_cols].copy()

# =========================================================
# 6. CHECK UNIQUE IDS
# =========================================================

print("\nUnique IDs:")
print("Environmental:", ENV["ID"].nunique())
print("Rainfall:", RAIN["ID"].nunique())
print("Satellite:", SAT["ID"].nunique())
print("Soil:", SOIL["ID"].nunique())

# =========================================================
# 7. MERGE RAINFALL
# =========================================================

FINAL = ENV.merge(
    RAIN,
    on="ID",
    how="left",
    validate="one_to_one"
)

print("\nAfter rainfall:", FINAL.shape)

# =========================================================
# 8. MERGE SATELLITE
# =========================================================

FINAL = FINAL.merge(
    SAT,
    on="ID",
    how="left",
    validate="one_to_one"
)

print("After satellite:", FINAL.shape)

# =========================================================
# 9. MERGE SOIL
# =========================================================

FINAL = FINAL.merge(
    SOIL,
    on="ID",
    how="left",
    validate="one_to_one"
)

print("After soil:", FINAL.shape)

# =========================================================
# 10. FEATURE GROUPS
# =========================================================

terrain = [
    "Elevation_m",
    "Slope_deg"
]

satellite = [
    "NDVI_mean",
    "NDVI_std",
    "NDVI_min",
    "NDVI_max",
    "NDVI_range",
    "Land_Cover_Type"
]

soil = [
    "Soil_clay_0_5cm",
    "Soil_sand_0_5cm",
    "Soil_silt_0_5cm",
    "Soil_soc_0_5cm",
    "Soil_bdod_0_5cm",
    "Soil_phh2o_0_5cm"
]

rainfall = [
    "rainfall_total_30d_mm",
    "rainfall_mean_daily_mm",
    "rainfall_max_daily_mm",
    "rainfall_std_daily_mm",
    "rainfall_7d_mm",
    "rainfall_15d_mm",
    "rainfall_30d_mm",
    "rainfall_max_7d_mm",
    "rainfall_max_15d_mm",
    "rainfall_max_30d_mm"
]

all_features = (
    terrain
    + satellite
    + soil
    + rainfall
)

# =========================================================
# 11. VALIDATION
# =========================================================

print("\n" + "=" * 70)
print("FINAL INTEGRATED DATASET")
print("=" * 70)

print("Rows:", len(FINAL))
print("Columns:", len(FINAL.columns))

print("\nClass distribution:")
print(
    FINAL["Landslide"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nFeature groups:")
print("Terrain:", len(terrain))
print("Satellite:", len(satellite))
print("Soil:", len(soil))
print("Rainfall:", len(rainfall))
print("Total model features:", len(all_features))

print("\nMissing values:")

for col in all_features:
    print(
        f"{col:35s} "
        f"{FINAL[col].isna().sum():5d}"
    )

# =========================================================
# 12. SAVE
# =========================================================

OUT = BASE / "slidronix_integrated_dataset_14270.csv"

FINAL.to_csv(
    OUT,
    index=False
)

print("\nSaved:")
print(OUT)

print("\n" + "=" * 70)
print("SUCCESS - INTEGRATED DATASET READY")
print("=" * 70)




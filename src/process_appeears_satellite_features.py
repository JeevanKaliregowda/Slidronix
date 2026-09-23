import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("data/raw/appeears/ndvi_lc_processed")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("SLIDRONIX - APPeEARS NDVI + LAND COVER PROCESSING")
print("=" * 70)

# =========================================================
# 1. NDVI
# =========================================================

ndvi_files = sorted(BASE.glob("*MOD13Q1-061-results.csv"))

print(f"\nNDVI files found: {len(ndvi_files)}")

if len(ndvi_files) != 15:
    raise RuntimeError(f"Expected 15 NDVI files, found {len(ndvi_files)}")

ndvi_parts = []

for f in ndvi_files:
    print("Reading:", f.name)

    df = pd.read_csv(f)

    keep = [
        "ID",
        "Latitude",
        "Longitude",
        "Date",
        "MOD13Q1_061__250m_16_days_NDVI",
        "MOD13Q1_061__250m_16_days_pixel_reliability"
    ]

    missing = [c for c in keep if c not in df.columns]

    if missing:
        raise ValueError(f"{f.name} missing columns: {missing}")

    ndvi_parts.append(df[keep])

ndvi = pd.concat(ndvi_parts, ignore_index=True)

print("\nRaw NDVI rows:", len(ndvi))
print("Unique NDVI IDs:", ndvi["ID"].nunique())

# ---------------------------------------------------------
# Convert date
# ---------------------------------------------------------

ndvi["Date"] = pd.to_datetime(ndvi["Date"])

# Remove 2021-12-19 observation.
# We want 2022 reference-year satellite features.
ndvi = ndvi[ndvi["Date"].dt.year == 2022].copy()

print("NDVI rows after removing 2021 observation:", len(ndvi))
print("NDVI IDs after filtering:", ndvi["ID"].nunique())

# ---------------------------------------------------------
# MODIS NDVI scaling
# Raw NDVI uses scale factor 0.0001.
# Fill value -3000 must NOT become real NDVI.
# ---------------------------------------------------------

raw = pd.to_numeric(
    ndvi["MOD13Q1_061__250m_16_days_NDVI"],
    errors="coerce"
)

raw[raw <= -3000] = np.nan

ndvi["NDVI"] = raw

# ---------------------------------------------------------
# Pixel reliability
#
# 0 = Good
# 1 = Marginal
# 2 = Snow/Ice
# 3 = Cloudy
#
# Keep 0 and 1.
# ---------------------------------------------------------

reliability = pd.to_numeric(
    ndvi[
        "MOD13Q1_061__250m_16_days_pixel_reliability"
    ],
    errors="coerce"
)

ndvi["NDVI_Reliability"] = reliability

ndvi.loc[
    ~ndvi["NDVI_Reliability"].isin([0, 1]),
    "NDVI"
] = np.nan

# ---------------------------------------------------------
# Aggregate annual NDVI
# ---------------------------------------------------------

ndvi_features = (
    ndvi.groupby("ID", as_index=False)
    .agg(
        NDVI_mean=("NDVI", "mean"),
        NDVI_std=("NDVI", "std"),
        NDVI_min=("NDVI", "min"),
        NDVI_max=("NDVI", "max"),
        NDVI_valid_observations=("NDVI", "count")
    )
)

ndvi_features["NDVI_range"] = (
    ndvi_features["NDVI_max"] -
    ndvi_features["NDVI_min"]
)

print("\nNDVI feature dataset:")
print("Rows:", len(ndvi_features))
print("Columns:", list(ndvi_features.columns))

print("\nNDVI missing values:")
print(ndvi_features.isna().sum().to_string())

print("\nNDVI statistics:")
print(
    ndvi_features[
        [
            "NDVI_mean",
            "NDVI_std",
            "NDVI_min",
            "NDVI_max",
            "NDVI_range",
            "NDVI_valid_observations"
        ]
    ].describe().to_string()
)

# =========================================================
# 2. LAND COVER
# =========================================================

lc_files = sorted(BASE.glob("*MCD12Q1-061-results.csv"))

print(f"\nLand Cover files found: {len(lc_files)}")

if len(lc_files) != 15:
    raise RuntimeError(
        f"Expected 15 Land Cover files, found {len(lc_files)}"
    )

lc_parts = []

for f in lc_files:
    print("Reading:", f.name)

    df = pd.read_csv(f)

    keep = [
        "ID",
        "Latitude",
        "Longitude",
        "Date",
        "MCD12Q1_061_LC_Type1",
        "MCD12Q1_061_QC"
    ]

    missing = [c for c in keep if c not in df.columns]

    if missing:
        raise ValueError(f"{f.name} missing columns: {missing}")

    lc_parts.append(df[keep])

lc = pd.concat(lc_parts, ignore_index=True)

print("\nRaw Land Cover rows:", len(lc))
print("Unique Land Cover IDs:", lc["ID"].nunique())

lc["Land_Cover_Type"] = pd.to_numeric(
    lc["MCD12Q1_061_LC_Type1"],
    errors="coerce"
)

lc["Land_Cover_QC"] = pd.to_numeric(
    lc["MCD12Q1_061_QC"],
    errors="coerce"
)

# One annual LC record per ID
lc = lc.drop_duplicates(subset="ID")

lc_features = lc[
    [
        "ID",
        "Land_Cover_Type",
        "Land_Cover_QC"
    ]
].copy()

print("\nLand Cover feature dataset:")
print("Rows:", len(lc_features))

print("\nLand Cover class distribution:")
print(
    lc_features["Land_Cover_Type"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nLand Cover missing values:")
print(lc_features.isna().sum().to_string())

# =========================================================
# 3. SAVE SATELLITE FEATURES
# =========================================================

ndvi_output = OUT / "ndvi_2022_features_14270.csv"
lc_output = OUT / "land_cover_2022_features_14270.csv"

ndvi_features.to_csv(ndvi_output, index=False)
lc_features.to_csv(lc_output, index=False)

print("\nSaved:")
print(ndvi_output)
print(lc_output)

# =========================================================
# 4. COMBINE NDVI + LAND COVER
# =========================================================

satellite = ndvi_features.merge(
    lc_features,
    on="ID",
    how="outer",
    validate="one_to_one"
)

print("\nCombined satellite dataset:")
print("Rows:", len(satellite))
print("Columns:", len(satellite.columns))

print("\nSatellite missing values:")
print(satellite.isna().sum().to_string())

satellite_output = OUT / "satellite_features_14270.csv"

satellite.to_csv(
    satellite_output,
    index=False
)

print("\nSaved:")
print(satellite_output)

# =========================================================
# 5. FINAL VALIDATION
# =========================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print("NDVI IDs:", len(ndvi_features))
print("Land Cover IDs:", len(lc_features))
print("Combined satellite IDs:", len(satellite))

if len(ndvi_features) != 14270:
    raise RuntimeError(
        f"Expected 14270 NDVI IDs, got {len(ndvi_features)}"
    )

if len(lc_features) != 14270:
    raise RuntimeError(
        f"Expected 14270 LC IDs, got {len(lc_features)}"
    )

if len(satellite) != 14270:
    raise RuntimeError(
        f"Expected 14270 combined IDs, got {len(satellite)}"
    )

print("\nSUCCESS: All 14,270 samples have NDVI + Land Cover records.")
print("=" * 70)

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from rasterio.windows import Window
from math import sqrt

INPUT = Path("data/processed/landslide_environmental_features.csv")
DEM_DIR = Path("data/raw/dem")
OUTPUT = Path("data/processed/landslide_environmental_features_repaired.csv")

df = pd.read_csv(INPUT)

missing = df["Elevation_m"].isna() | df["Slope_deg"].isna()

print("=" * 65)
print("REPAIRING MISSING DEM FEATURES")
print("=" * 65)
print(f"Total samples: {len(df):,}")
print(f"Samples needing repair: {missing.sum():,}")

# Cache opened DEM files
dem_cache = {}

def get_dem_file(lat, lon):
    lat_deg = int(np.floor(lat))
    lon_deg = int(np.floor(lon))

    lat_str = f"N{lat_deg:02d}" if lat_deg >= 0 else f"S{-lat_deg:02d}"
    lon_str = f"E{lon_deg:03d}" if lon_deg >= 0 else f"W{-lon_deg:03d}"

    pattern = f"*{lat_str}*{lon_str}*.tif"
    matches = list(DEM_DIR.glob(pattern))

    if not matches:
        return None

    return matches[0]


def find_valid_pixel(src, lat, lon, radius=10):
    """
    Find the nearest valid DEM pixel around the requested coordinate.
    """

    try:
        row, col = src.index(lon, lat)
    except Exception:
        return None

    size = radius * 2 + 1

    window = Window(
        col - radius,
        row - radius,
        size,
        size
    )

    try:
        arr = src.read(
            1,
            window=window,
            boundless=True,
            fill_value=np.nan
        ).astype(float)
    except Exception:
        return None

    valid = np.isfinite(arr)

    if not valid.any():
        return None

    # Find nearest valid pixel to center
    center = radius
    positions = np.argwhere(valid)

    distances = (
        (positions[:, 0] - center) ** 2 +
        (positions[:, 1] - center) ** 2
    )

    idx = np.argmin(distances)

    rr, cc = positions[idx]

    return float(arr[rr, cc]), row, col


def calculate_local_slope(src, row, col, radius=2):
    """
    Calculate slope from a small local DEM neighborhood.
    """

    window = Window(
        col - radius,
        row - radius,
        radius * 2 + 1,
        radius * 2 + 1
    )

    try:
        elev = src.read(
            1,
            window=window,
            boundless=True,
            fill_value=np.nan
        ).astype(float)
    except Exception:
        return np.nan

    if not np.isfinite(elev).all():
        # Replace local missing pixels with nearest valid value
        valid = elev[np.isfinite(elev)]

        if len(valid) < 9:
            return np.nan

        elev = np.where(
            np.isfinite(elev),
            elev,
            np.nanmedian(elev)
        )

    # Pixel dimensions in metres
    transform = src.window_transform(window)

    pixel_x = abs(transform.a)
    pixel_y = abs(transform.e)

    if pixel_x <= 0 or pixel_y <= 0:
        return np.nan

    # Horn-style gradient
    dz_dy, dz_dx = np.gradient(
        elev,
        pixel_y,
        pixel_x
    )

    center = radius

    gradient = sqrt(
        dz_dx[center, center] ** 2 +
        dz_dy[center, center] ** 2
    )

    slope = np.degrees(np.arctan(gradient))

    return float(slope)


repaired = 0
failed = 0

for count, idx in enumerate(df.index[missing], 1):

    lat = float(df.at[idx, "Latitude"])
    lon = float(df.at[idx, "Longitude"])

    dem_file = get_dem_file(lat, lon)

    if dem_file is None:
        failed += 1
        continue

    dem_key = str(dem_file)

    if dem_key not in dem_cache:
        try:
            dem_cache[dem_key] = rasterio.open(dem_file)
        except Exception:
            failed += 1
            continue

    src = dem_cache[dem_key]

    result = find_valid_pixel(
        src,
        lat,
        lon,
        radius=10
    )

    if result is None:
        failed += 1
        continue

    elevation, row, col = result

    slope = calculate_local_slope(
        src,
        row,
        col,
        radius=2
    )

    if np.isfinite(elevation):
        df.at[idx, "Elevation_m"] = elevation

    if np.isfinite(slope):
        df.at[idx, "Slope_deg"] = slope

    if (
        np.isfinite(df.at[idx, "Elevation_m"])
        and np.isfinite(df.at[idx, "Slope_deg"])
    ):
        repaired += 1
    else:
        failed += 1

    if count % 100 == 0 or count == missing.sum():
        print(
            f"[{count:,}/{missing.sum():,}] "
            f"Repaired: {repaired:,} | Failed: {failed:,}"
        )

for src in dem_cache.values():
    src.close()

remaining = df["Elevation_m"].isna() | df["Slope_deg"].isna()

print("\n" + "=" * 65)
print("REPAIR COMPLETE")
print("=" * 65)
print(f"Originally missing : {missing.sum():,}")
print(f"Successfully repaired: {repaired:,}")
print(f"Still missing      : {remaining.sum():,}")

df.to_csv(OUTPUT, index=False)

print(f"\nSaved repaired dataset:")
print(OUTPUT)

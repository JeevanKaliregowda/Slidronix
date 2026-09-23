import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from rasterio.windows import Window

INPUT = Path("data/processed/landslide_environmental_features.csv")
DEM_DIR = Path("data/raw/dem")

df = pd.read_csv(INPUT)
missing = df["Elevation_m"].isna()

def get_dem_file(lat, lon):
    lat_deg = int(np.floor(lat))
    lon_deg = int(np.floor(lon))

    lat_str = f"N{lat_deg:02d}" if lat_deg >= 0 else f"S{-lat_deg:02d}"
    lon_str = f"E{lon_deg:03d}" if lon_deg >= 0 else f"W{-lon_deg:03d}"

    matches = list(DEM_DIR.glob(f"*{lat_str}*{lon_str}*.tif"))
    return matches[0] if matches else None

cache = {}

distances = []
no_valid = 0

for n, idx in enumerate(df.index[missing], 1):
    lat = float(df.at[idx, "Latitude"])
    lon = float(df.at[idx, "Longitude"])

    path = get_dem_file(lat, lon)

    if path is None:
        no_valid += 1
        continue

    key = str(path)

    if key not in cache:
        cache[key] = rasterio.open(path)

    src = cache[key]

    try:
        row, col = src.index(lon, lat)

        radius = 100
        size = radius * 2 + 1

        arr = src.read(
            1,
            window=Window(col-radius, row-radius, size, size),
            boundless=True,
            fill_value=np.nan
        ).astype(float)

        valid = np.isfinite(arr)

        if valid.any():
            center = radius
            pos = np.argwhere(valid)

            d = np.sqrt(
                (pos[:,0] - center)**2 +
                (pos[:,1] - center)**2
            )

            distances.append(float(d.min()))
        else:
            no_valid += 1

    except Exception:
        no_valid += 1

    if n % 250 == 0:
        print(f"Checked {n:,}/{missing.sum():,}")

for src in cache.values():
    src.close()

distances = np.array(distances)

print("\n" + "="*65)
print("DEM NEAREST-VALID-PIXEL DIAGNOSTIC")
print("="*65)

print(f"Missing samples examined : {missing.sum():,}")
print(f"With valid pixel nearby  : {len(distances):,}")
print(f"No valid pixel in 100px  : {no_valid:,}")

if len(distances):
    print("\nDistance to nearest valid pixel:")
    print(f"Minimum : {distances.min():.1f} pixels")
    print(f"Median  : {np.median(distances):.1f} pixels")
    print(f"Mean    : {distances.mean():.1f} pixels")
    print(f"90%     : {np.percentile(distances,90):.1f} pixels")
    print(f"95%     : {np.percentile(distances,95):.1f} pixels")
    print(f"Maximum : {distances.max():.1f} pixels")

    print("\nApproximate distance assuming ~30 m DEM pixels:")
    print(f"Median  : {np.median(distances)*30:.0f} m")
    print(f"90%     : {np.percentile(distances,90)*30:.0f} m")
    print(f"95%     : {np.percentile(distances,95)*30:.0f} m")
    print(f"Maximum : {distances.max()*30:.0f} m")

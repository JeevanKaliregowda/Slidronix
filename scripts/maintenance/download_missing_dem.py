import pandas as pd
import urllib.request
from pathlib import Path

INPUT = Path("data/processed/landslide_binary_dataset.csv")
DEM_DIR = Path("data/raw/dem")
DEM_DIR.mkdir(parents=True, exist_ok=True)

AWS_S3_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

df = pd.read_csv(INPUT)

lats = df["Latitude"].astype(float).apply(lambda x: int(x // 1))
lons = df["Longitude"].astype(float).apply(lambda x: int(x // 1))

unique_tile_coords = sorted(set(zip(lats, lons)))

tiles = []

for lat_deg, lon_deg in unique_tile_coords:
    lat_str = f"N{lat_deg:02d}" if lat_deg >= 0 else f"S{-lat_deg:02d}"
    lon_str = f"E{lon_deg:03d}" if lon_deg >= 0 else f"W{-lon_deg:03d}"

    tile_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"

    tiles.append({
        "tile_name": tile_name,
        "url": f"{AWS_S3_BASE_URL}/{tile_name}/{tile_name}.tif",
        "path": DEM_DIR / f"{tile_name}.tif"
    })

missing = [t for t in tiles if not t["path"].exists()]

print(f"Required tiles : {len(tiles)}")
print(f"Already local  : {len(tiles) - len(missing)}")
print(f"Missing        : {len(missing)}")
print()

for i, tile in enumerate(missing, 1):
    print(f"[{i}/{len(missing)}] Downloading:")
    print(f"  {tile['tile_name']}")

    try:
        urllib.request.urlretrieve(tile["url"], tile["path"])
        size = tile["path"].stat().st_size
        print(f"  OK - {size:,} bytes")
    except Exception as e:
        print(f"  FAILED - {e}")
        if tile["path"].exists():
            tile["path"].unlink()

print("\nDownload phase complete.")

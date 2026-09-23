import os
import time
import requests
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

POINTS_FILE = "data/processed/appeears_points_14270.csv"
OUTPUT_FILE = "data/processed/rainfall/chirps_2022_daily_14270.csv"
TEMP_FILE = "data/raw/rainfall/chirps_temp.tif"

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat/2022"

WEST, SOUTH, EAST, NORTH = 72, 8, 79, 21

os.makedirs("data/processed/rainfall", exist_ok=True)
os.makedirs("data/raw/rainfall", exist_ok=True)

points = pd.read_csv(POINTS_FILE)

print("Points:", len(points))

# Prepare output
result = points[["ID", "Latitude", "Longitude"]].copy()

for day in pd.date_range("2022-01-01", "2022-01-30", freq="D"):

    date_str = day.strftime("%Y.%m.%d")
    column = day.strftime("%Y-%m-%d")

    filename = f"chirps-v3.0.sat.{date_str}.tif"
    url = f"{BASE_URL}/{filename}"

    print(f"\n[{column}] Downloading...")

    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()

        with open(TEMP_FILE, "wb") as f:
            f.write(r.content)

        with rasterio.open(TEMP_FILE) as src:

            window = from_bounds(
                WEST, SOUTH, EAST, NORTH,
                src.transform
            )

            rainfall = src.read(1, window=window)
            transform = src.window_transform(window)

            values = []

            for lat, lon in zip(
                points["Latitude"],
                points["Longitude"]
            ):
                col, row = ~transform * (lon, lat)

                row = int(round(row))
                col = int(round(col))

                if (
                    0 <= row < rainfall.shape[0]
                    and 0 <= col < rainfall.shape[1]
                ):
                    value = float(rainfall[row, col])
                else:
                    value = float("nan")

                values.append(value)

        result[column] = values

        print(
            f"  OK | Values: {len(values)} | "
            f"Min: {pd.Series(values).min():.2f} | "
            f"Max: {pd.Series(values).max():.2f}"
        )

        os.remove(TEMP_FILE)

    except Exception as e:
        print(f"  ERROR: {e}")

        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)

        result[column] = float("nan")

    # Save progress after every day
    result.to_csv(OUTPUT_FILE, index=False)

print("\n===================================")
print("RAINFALL EXTRACTION COMPLETE")
print("===================================")
print("Rows:", len(result))
print("Columns:", len(result.columns))
print("Output:", OUTPUT_FILE)

import pandas as pd
import numpy as np

src = "data/processed/rainfall/chirps_2022_daily_14270.csv"
out = "data/processed/rainfall/chirps_2022_30day_features_14270.csv"

df = pd.read_csv(src)

dates = pd.date_range("2022-01-01", "2022-01-30", freq="D")
rain_cols = [d.strftime("%Y-%m-%d") for d in dates]

rain = df[rain_cols]

features = df[["ID", "Latitude", "Longitude"]].copy()

features["rainfall_total_30d_mm"] = rain.sum(axis=1)
features["rainfall_mean_daily_mm"] = rain.mean(axis=1)
features["rainfall_max_daily_mm"] = rain.max(axis=1)
features["rainfall_std_daily_mm"] = rain.std(axis=1)

features["rainfall_7d_mm"] = rain.iloc[:, -7:].sum(axis=1)
features["rainfall_15d_mm"] = rain.iloc[:, -15:].sum(axis=1)
features["rainfall_30d_mm"] = rain.sum(axis=1)

features["rainfall_max_7d_mm"] = rain.iloc[:, -7:].max(axis=1)
features["rainfall_max_15d_mm"] = rain.iloc[:, -15:].max(axis=1)
features["rainfall_max_30d_mm"] = rain.max(axis=1)

features.to_csv(out, index=False)

print("Created:", out)
print("Shape:", features.shape)
print()
print(features.describe().round(3).to_string())

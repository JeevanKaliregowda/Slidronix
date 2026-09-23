import pandas as pd
import numpy as np
import requests
import rasterio
from rasterio.io import MemoryFile
from pathlib import Path

BASE = Path("data/processed")
ENV = BASE / "landslide_environmental_features_clean.csv"
OUT = BASE / "soil_features_14270.csv"

df = pd.read_csv(ENV)

print("=" * 70)
print("SLIDRONIX - SOILGRIDS EXTRACTION")
print("=" * 70)
print("Samples:", len(df))

# Bounding box around all study samples
minx = df["Longitude"].min()
maxx = df["Longitude"].max()
miny = df["Latitude"].min()
maxy = df["Latitude"].max()

print("Bounding box:")
print(minx, maxx, miny, maxy)

# SoilGrids WCS
BASE_URL = "https://maps.isric.org/mapserv"

# Property -> SoilGrids map
properties = {
    "clay": "clay",
    "sand": "sand",
    "silt": "silt",
    "soc": "soc",
    "bdod": "bdod",
    "phh2o": "phh2o"
}

# 0-5 cm median
depth = "0-5cm"
quantile = "Q0.5"

result = df[["S.No", "Latitude", "Longitude"]].copy()

for name, mapname in properties.items():

    coverage = f"{name}_{depth}_{quantile}"

    print("\n" + "-" * 60)
    print("Downloading:", coverage)

    params = {
        "map": f"/map/{mapname}.map",
        "SERVICE": "WCS",
        "VERSION": "2.0.1",
        "REQUEST": "GetCoverage",
        "COVERAGEID": coverage,
        "FORMAT": "GEOTIFF_INT16",
        "SUBSET": [
            f"X({minx},{maxx})",
            f"Y({miny},{maxy})"
        ],
        "SUBSETTINGCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
        "OUTPUTCRS": "http://www.opengis.net/def/crs/EPSG/0/4326"
    }

    try:
        r = requests.get(
            BASE_URL,
            params=params,
            timeout=180
        )

        print("HTTP:", r.status_code)
        print("Size MB:", round(len(r.content) / 1024 / 1024, 2))

        r.raise_for_status()

        with MemoryFile(r.content) as memfile:
            with memfile.open() as src:

                print("Raster:", src.width, "x", src.height)
                print("CRS:", src.crs)

                coords = list(
                    zip(
                        df["Longitude"],
                        df["Latitude"]
                    )
                )

                values = []

                for val in src.sample(coords):
                    v = val[0]

                    if src.nodata is not None and v == src.nodata:
                        values.append(np.nan)
                    else:
                        values.append(float(v))

                values = np.array(values)

                # SoilGrids integer scaling:
                # convert according to variable.
                if name in ["clay", "sand", "silt"]:
                    # SoilGrids stores g/kg with factor 10
                    values = values / 10.0

                elif name == "soc":
                    # dg/kg with factor 10 -> g/kg
                    values = values / 10.0

                elif name == "bdod":
                    # cg/cm3 with factor 100 -> g/cm3
                    values = values / 100.0

                elif name == "phh2o":
                    # pH x 10
                    values = values / 10.0

                result[f"Soil_{name}_0_5cm"] = values

                print(
                    "Valid:",
                    np.sum(~np.isnan(values)),
                    "/",
                    len(values)
                )

    except Exception as e:
        print("ERROR:", repr(e))
        result[f"Soil_{name}_0_5cm"] = np.nan

result.to_csv(OUT, index=False)

print("\n" + "=" * 70)
print("SOIL EXTRACTION COMPLETE")
print("=" * 70)

print("Saved:", OUT)
print("\nMissing values:")
print(result.isna().sum().to_string())

print("\nStatistics:")
print(result.describe().to_string())

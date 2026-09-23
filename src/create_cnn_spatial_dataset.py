import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


# ============================================================
# Slidronix - CNN Spatial Dataset Generator
# ============================================================
#
# Creates real spatial terrain patches from Copernicus DEM.
#
# Channels:
#   Channel 0 -> Elevation
#   Channel 1 -> Slope
#
# Patch:
#   32 x 32 pixels
#
# The Phase E train/validation/test split is preserved.
# ============================================================


INPUT_DIR = "data/processed/phase_e"
DEM_DIR = "data/raw/dem"
OUTPUT_DIR = "data/processed/phase_f/cnn_spatial"

PATCH_SIZE = 32
HALF = PATCH_SIZE // 2

TRAIN_FILE = os.path.join(
    INPUT_DIR,
    "slidronix_phase_e_train.csv"
)

VAL_FILE = os.path.join(
    INPUT_DIR,
    "slidronix_phase_e_validation.csv"
)

TEST_FILE = os.path.join(
    INPUT_DIR,
    "slidronix_phase_e_test.csv"
)


os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_dem_file(lat, lon):
    """
    Find the DEM tile containing a geographic coordinate.
    """

    for path in DEM_FILES:

        with rasterio.open(path) as src:

            bounds = src.bounds

            if (
                bounds.left <= lon <= bounds.right
                and
                bounds.bottom <= lat <= bounds.top
            ):
                return path

    return None


def calculate_slope(elevation, transform):
    """
    Calculate slope in degrees using the DEM patch.

    np.gradient gives elevation change per pixel.
    Pixel dimensions are converted approximately
    from degrees to metres using the latitude.
    """

    # Mean latitude of patch
    lat_center = CURRENT_LAT

    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = (
        111320.0 * np.cos(np.radians(lat_center))
    )

    pixel_width_m = abs(transform.a) * meters_per_degree_lon
    pixel_height_m = abs(transform.e) * meters_per_degree_lat

    dz_dy, dz_dx = np.gradient(
        elevation,
        pixel_height_m,
        pixel_width_m
    )

    slope_rad = np.arctan(
        np.sqrt(
            dz_dx ** 2 +
            dz_dy ** 2
        )
    )

    slope_deg = np.degrees(slope_rad)

    return slope_deg


def extract_patch(lat, lon):
    """
    Extract a 32x32 elevation + slope patch.
    """

    global CURRENT_LAT

    CURRENT_LAT = lat

    dem_path = find_dem_file(lat, lon)

    if dem_path is None:
        return None

    with rasterio.open(dem_path) as src:

        row, col = src.index(lon, lat)

        window = Window(
            col - HALF,
            row - HALF,
            PATCH_SIZE,
            PATCH_SIZE
        )

        elevation = src.read(
            1,
            window=window,
            boundless=True,
            fill_value=np.nan
        ).astype(np.float32)

        # Make sure the patch has the correct dimensions
        if elevation.shape != (
            PATCH_SIZE,
            PATCH_SIZE
        ):
            return None

        # Handle DEM nodata values
        if src.nodata is not None:
            elevation[
                elevation == src.nodata
            ] = np.nan

        # Check valid pixels
        valid_ratio = np.isfinite(
            elevation
        ).mean()

        if valid_ratio < 0.95:
            return None

        # Fill small missing areas using patch median
        median_value = np.nanmedian(elevation)

        elevation = np.where(
            np.isfinite(elevation),
            elevation,
            median_value
        )

        # Transform corresponding to patch
        patch_transform = src.window_transform(window)

        slope = calculate_slope(
            elevation,
            patch_transform
        )

        # Replace invalid slope values
        slope = np.nan_to_num(
            slope,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Final tensor:
        # [2, 32, 32]
        patch = np.stack(
            [
                elevation,
                slope
            ],
            axis=0
        ).astype(np.float32)

        return patch


def process_split(name, filepath):

    print("\n" + "=" * 60)
    print("Processing:", name)
    print("=" * 60)

    df = pd.read_csv(filepath)

    print("Samples:", len(df))

    patches = []
    labels = []
    ids = []

    failed = 0

    for i, row in df.iterrows():

        patch = extract_patch(
            float(row["Latitude"]),
            float(row["Longitude"])
        )

        if patch is None:
            failed += 1
            continue

        patches.append(patch)

        labels.append(
            int(row["Landslide"])
        )

        ids.append(
            row["ID"]
        )

        if (i + 1) % 500 == 0:

            print(
                f"Processed {i + 1}/{len(df)}"
            )

    if len(patches) == 0:

        raise RuntimeError(
            f"No valid patches generated for {name}"
        )

    X = np.stack(
        patches
    ).astype(np.float32)

    y = np.array(
        labels,
        dtype=np.int64
    )

    ids = np.array(
        ids,
        dtype=str
    )

    # --------------------------------------------------------
    # Save spatial tensors
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{name}_cnn_patches.npz"
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        ids=ids
    )

    print("\nCompleted:", name)
    print("Valid patches:", len(X))
    print("Failed patches:", failed)
    print("Tensor shape:", X.shape)

    print(
        "Landslide:",
        int((y == 1).sum())
    )

    print(
        "Background:",
        int((y == 0).sum())
    )

    print(
        "Saved:",
        output_file
    )


# ============================================================
# Locate DEM tiles once
# ============================================================

DEM_FILES = glob.glob(
    os.path.join(
        DEM_DIR,
        "*.tif"
    )
)

print("DEM tiles found:", len(DEM_FILES))

if len(DEM_FILES) == 0:

    raise RuntimeError(
        "No DEM tiles found."
    )


# ============================================================
# Process the three existing Phase E splits
# ============================================================

process_split(
    "train",
    TRAIN_FILE
)

process_split(
    "validation",
    VAL_FILE
)

process_split(
    "test",
    TEST_FILE
)


print("\n" + "=" * 60)
print("CNN SPATIAL DATASET COMPLETE")
print("=" * 60)
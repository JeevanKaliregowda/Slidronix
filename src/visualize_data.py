import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landslide_inventory_raw.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "charts"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("LANDSLIDE DATA VISUALIZATION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


# --------------------------------------------------
# CLEAN COORDINATES
# --------------------------------------------------

df["Latitude"] = pd.to_numeric(
    df["Latitude"],
    errors="coerce"
)

df["Longitude"] = pd.to_numeric(
    df["Longitude"],
    errors="coerce"
)

geo_df = df.dropna(
    subset=["Latitude", "Longitude"]
)


# --------------------------------------------------
# 1. LANDSLIDES BY STATE
# --------------------------------------------------

state_counts = (
    df["State"]
    .value_counts()
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))

state_counts.plot(
    kind="barh"
)

plt.title(
    "Top 10 States by Recorded Landslides",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Number of Recorded Landslides")
plt.ylabel("State")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "landslides_by_state.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 2. MOVEMENT TYPE DISTRIBUTION
#    TOP TYPES + OTHER
# --------------------------------------------------

movement_counts = df["Movement Type"].value_counts()

# Keep only top 8 movement types
top_movement = movement_counts.head(8).copy()

# Combine remaining types into "Other"
other_count = movement_counts.iloc[8:].sum()

if other_count > 0:
    top_movement["Other Movement Types"] = other_count

# Sort for horizontal chart
top_movement = top_movement.sort_values()

plt.figure(figsize=(10, 6))

top_movement.plot(
    kind="barh"
)

plt.title(
    "Major Landslide Movement Types",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Number of Recorded Landslides")
plt.ylabel("Movement Type")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "movement_type_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 3. ALL INDIA GEOGRAPHICAL DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 10))

plt.scatter(
    geo_df["Longitude"],
    geo_df["Latitude"],
    s=4,
    alpha=0.35
)

plt.title(
    "Geographical Distribution of Recorded Landslides in India",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "landslide_geographical_distribution_india.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 4. WESTERN GHATS FOCUSED VISUALIZATION
# --------------------------------------------------

# Approximate geographic focus region of the Western Ghats
western_ghats = geo_df[
    (geo_df["Latitude"] >= 8) &
    (geo_df["Latitude"] <= 22) &
    (geo_df["Longitude"] >= 72.5) &
    (geo_df["Longitude"] <= 77.5)
]

print(
    f"\nRecords in approximate Western Ghats region: "
    f"{len(western_ghats)}"
)


plt.figure(figsize=(8, 10))

plt.scatter(
    western_ghats["Longitude"],
    western_ghats["Latitude"],
    s=6,
    alpha=0.5
)

plt.title(
    "Recorded Landslide Distribution: Western Ghats Region",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "western_ghats_landslide_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 5. WESTERN GHATS STATE DISTRIBUTION
# --------------------------------------------------

western_ghats_states = [
    "Kerala",
    "Karnataka",
    "Tamil Nadu",
    "Maharashtra",
    "Goa"
]

wg_state_df = df[
    df["State"].isin(western_ghats_states)
]

wg_state_counts = (
    wg_state_df["State"]
    .value_counts()
    .sort_values()
)


plt.figure(figsize=(9, 6))

wg_state_counts.plot(
    kind="barh"
)

plt.title(
    "Recorded Landslides in Western Ghats States",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Number of Recorded Landslides")
plt.ylabel("State")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "western_ghats_states_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated charts:")

for file in OUTPUT_DIR.glob("*.png"):
    print("-", file.name)
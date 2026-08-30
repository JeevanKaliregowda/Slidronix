import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landslide_inventory_raw.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landslide_inventory_cleaned.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("LANDSLIDE DATA PREPROCESSING")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print(f"\nOriginal dataset shape: {df.shape}")


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


# Remove invalid coordinates
df = df[
    df["Latitude"].between(6, 38) &
    df["Longitude"].between(68, 98)
]


# --------------------------------------------------
# CLEAN TEXT COLUMNS
# --------------------------------------------------

text_columns = [
    "State",
    "District",
    "Subdivision Or Taluk",
    "Material Involved",
    "Movement Type"
]

for column in text_columns:
    df[column] = df[column].astype("string").str.strip()


# --------------------------------------------------
# CLEAN INITIATION YEAR
# --------------------------------------------------

df["Initiation_Year"] = pd.to_numeric(
    df["Initiation_Year"],
    errors="coerce"
)

# Replace 0 with missing value
df.loc[df["Initiation_Year"] == 0, "Initiation_Year"] = pd.NA


# --------------------------------------------------
# REMOVE DUPLICATES
# --------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print(
    f"\nDuplicates removed: "
    f"{before_duplicates - after_duplicates}"
)


# --------------------------------------------------
# REMOVE RECORDS WITHOUT TARGET
# --------------------------------------------------

df = df.dropna(
    subset=[
        "Movement Type",
        "State",
        "Material Involved"
    ]
)


# --------------------------------------------------
# SAVE CLEANED DATA
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(f"\nFinal dataset shape: {df.shape}")

print("\nMissing values:")
print(df.isnull().sum())

print("\nTop Movement Types:")
print(df["Movement Type"].value_counts().head(10))

print(f"\nCleaned dataset saved to:")
print(OUTPUT_PATH)
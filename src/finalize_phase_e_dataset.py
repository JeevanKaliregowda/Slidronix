from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# Slidronix - Phase E Dataset Finalization
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "slidronix_integrated_dataset_14270.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase_e"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BALANCED_OUTPUT = OUTPUT_DIR / "slidronix_phase_e_balanced.csv"
TRAIN_OUTPUT = OUTPUT_DIR / "slidronix_phase_e_train.csv"
VAL_OUTPUT = OUTPUT_DIR / "slidronix_phase_e_validation.csv"
TEST_OUTPUT = OUTPUT_DIR / "slidronix_phase_e_test.csv"
SUMMARY_OUTPUT = ROOT / "outputs" / "phase_e_final_summary.txt"


# ------------------------------------------------------------
# 1. Load integrated dataset
# ------------------------------------------------------------

print("=" * 70)
print("SLIDRONIX - PHASE E FINALIZATION")
print("=" * 70)

df = pd.read_csv(INPUT)

print(f"\nInput dataset: {df.shape}")
print("\nOriginal class distribution:")
print(df["Sample_Type"].value_counts().sort_index())


# ------------------------------------------------------------
# 2. Basic validation
# ------------------------------------------------------------

required_columns = ["ID", "Latitude", "Longitude", "Sample_Type"]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )

df = df.dropna(
    subset=["Latitude", "Longitude", "Sample_Type"]
).copy()

label_map = {
    "Background": 0,
    "Landslide": 1
}

df["Sample_Type"] = (
    df["Sample_Type"]
    .astype(str)
    .str.strip()
    .map(label_map)
)

if df["Sample_Type"].isna().any():
    raise ValueError(
        "Unexpected Sample_Type values found: "
        + str(df["Sample_Type"].unique())
    )

df["Sample_Type"] = df["Sample_Type"].astype(int)

# ------------------------------------------------------------
# 3. Balance the dataset
# ------------------------------------------------------------

positive = df[df["Sample_Type"] == 1].copy()
background = df[df["Sample_Type"] == 0].copy()

print("\nBefore balancing:")
print(f"Landslide : {len(positive)}")
print(f"Background: {len(background)}")

target_count = min(len(positive), len(background))

print(f"\nBalanced target per class: {target_count}")

# Fixed random seed = reproducible experiment
background_balanced = background.sample(
    n=target_count,
    random_state=42
)

positive_balanced = positive.sample(
    n=target_count,
    random_state=42
)

balanced = pd.concat(
    [positive_balanced, background_balanced],
    ignore_index=True
)

# Shuffle final balanced dataset
balanced = balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ------------------------------------------------------------
# 4. Create spatial blocks
# ------------------------------------------------------------

# Approximately 0.1 degree spatial blocks.
# This is roughly 10-11 km in the Western Ghats region.

BLOCK_SIZE = 0.1

balanced["spatial_block_lat"] = np.floor(
    balanced["Latitude"] / BLOCK_SIZE
).astype(int)

balanced["spatial_block_lon"] = np.floor(
    balanced["Longitude"] / BLOCK_SIZE
).astype(int)

balanced["spatial_block"] = (
    balanced["spatial_block_lat"].astype(str)
    + "_"
    + balanced["spatial_block_lon"].astype(str)
)


# ------------------------------------------------------------
# 5. Verify duplicate coordinates
# ------------------------------------------------------------

duplicate_coordinates = balanced.duplicated(
    subset=["Latitude", "Longitude"],
    keep=False
).sum()

print(
    f"\nRows involved in duplicate coordinates: "
    f"{duplicate_coordinates}"
)


# ------------------------------------------------------------
# 6. Spatial train/validation/test split
# ------------------------------------------------------------

X = balanced.drop(columns=["Sample_Type"])
y = balanced["Sample_Type"]

groups = balanced["spatial_block"]

print(
    f"Unique spatial blocks: "
    f"{groups.nunique()}"
)

# Five spatial folds.
# Fold 0 -> test
# Fold 1 -> validation
# Remaining folds -> training

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_assignments = np.full(
    len(balanced),
    -1,
    dtype=int
)

for fold_number, (_, test_indices) in enumerate(
    sgkf.split(X, y, groups)
):
    fold_assignments[test_indices] = fold_number

balanced["spatial_fold"] = fold_assignments

if (balanced["spatial_fold"] < 0).any():
    raise RuntimeError(
        "Some rows were not assigned to a spatial fold."
    )

test = balanced[
    balanced["spatial_fold"] == 0
].copy()

validation = balanced[
    balanced["spatial_fold"] == 1
].copy()

train = balanced[
    balanced["spatial_fold"].isin([2, 3, 4])
].copy()


# ------------------------------------------------------------
# 7. Remove helper columns from final modelling datasets
# ------------------------------------------------------------

helper_columns = [
    "spatial_block_lat",
    "spatial_block_lon",
    "spatial_block",
    "spatial_fold"
]

balanced_clean = balanced.drop(
    columns=helper_columns
)

train_clean = train.drop(
    columns=helper_columns
)

validation_clean = validation.drop(
    columns=helper_columns
)

test_clean = test.drop(
    columns=helper_columns
)


# ------------------------------------------------------------
# 8. Verify spatial leakage
# ------------------------------------------------------------

train_blocks = set(train["spatial_block"])
val_blocks = set(validation["spatial_block"])
test_blocks = set(test["spatial_block"])

train_val_overlap = train_blocks & val_blocks
train_test_overlap = train_blocks & test_blocks
val_test_overlap = val_blocks & test_blocks

print("\nSpatial leakage check:")
print(
    f"Train ∩ Validation blocks: "
    f"{len(train_val_overlap)}"
)
print(
    f"Train ∩ Test blocks: "
    f"{len(train_test_overlap)}"
)
print(
    f"Validation ∩ Test blocks: "
    f"{len(val_test_overlap)}"
)

if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):
    raise RuntimeError(
        "Spatial leakage detected!"
    )


# ------------------------------------------------------------
# 9. Save datasets
# ------------------------------------------------------------

balanced_clean.to_csv(
    BALANCED_OUTPUT,
    index=False
)

train_clean.to_csv(
    TRAIN_OUTPUT,
    index=False
)

validation_clean.to_csv(
    VAL_OUTPUT,
    index=False
)

test_clean.to_csv(
    TEST_OUTPUT,
    index=False
)


# ------------------------------------------------------------
# 10. Generate summary
# ------------------------------------------------------------

def class_counts(data):
    counts = data["Sample_Type"].value_counts().sort_index()

    return (
        f"Background (0): {counts.get(0, 0)}\n"
        f"Landslide  (1): {counts.get(1, 0)}"
    )


summary = []

summary.append("SLIDRONIX - PHASE E FINAL DATASET SUMMARY")
summary.append("=" * 70)
summary.append("")

summary.append("INPUT")
summary.append("-" * 70)
summary.append(f"File: {INPUT}")
summary.append(f"Rows: {len(df)}")
summary.append(f"Columns: {len(df.columns)}")
summary.append("")

summary.append("BALANCING")
summary.append("-" * 70)
summary.append(
    "Balanced by random undersampling of the background class."
)
summary.append("Random seed: 42")
summary.append(f"Final balanced rows: {len(balanced_clean)}")
summary.append(class_counts(balanced_clean))
summary.append("")

summary.append("SPATIAL BLOCKING")
summary.append("-" * 70)
summary.append(
    "Spatial block size: 0.1° x 0.1°"
)
summary.append(
    "Approximate scale in study region: ~10-11 km"
)
summary.append(
    f"Unique spatial blocks: {groups.nunique()}"
)
summary.append("")

summary.append("SPATIAL SPLIT")
summary.append("-" * 70)
summary.append(
    "StratifiedGroupKFold with 5 spatial folds."
)
summary.append(
    "Fold 0 = test"
)
summary.append(
    "Fold 1 = validation"
)
summary.append(
    "Folds 2-4 = training"
)
summary.append("")

summary.append("TRAINING SET")
summary.append("-" * 70)
summary.append(f"Rows: {len(train_clean)}")
summary.append(class_counts(train_clean))
summary.append(
    f"Spatial blocks: {len(train_blocks)}"
)
summary.append("")

summary.append("VALIDATION SET")
summary.append("-" * 70)
summary.append(f"Rows: {len(validation_clean)}")
summary.append(class_counts(validation_clean))
summary.append(
    f"Spatial blocks: {len(val_blocks)}"
)
summary.append("")

summary.append("TEST SET")
summary.append("-" * 70)
summary.append(f"Rows: {len(test_clean)}")
summary.append(class_counts(test_clean))
summary.append(
    f"Spatial blocks: {len(test_blocks)}"
)
summary.append("")

summary.append("LEAKAGE CHECK")
summary.append("-" * 70)
summary.append(
    f"Train/Validation block overlap: "
    f"{len(train_val_overlap)}"
)
summary.append(
    f"Train/Test block overlap: "
    f"{len(train_test_overlap)}"
)
summary.append(
    f"Validation/Test block overlap: "
    f"{len(val_test_overlap)}"
)
summary.append("")

summary.append("OUTPUT FILES")
summary.append("-" * 70)
summary.append(str(BALANCED_OUTPUT))
summary.append(str(TRAIN_OUTPUT))
summary.append(str(VAL_OUTPUT))
summary.append(str(TEST_OUTPUT))
summary.append("")

summary.append("PHASE E STATUS")
summary.append("-" * 70)
summary.append(
    "Phase E dataset construction completed."
)
summary.append(
    "The final modelling data is class-balanced and "
    "spatially partitioned to reduce geographic leakage."
)
summary.append(
    "Phase F can use the train/validation/test files "
    "as the fixed modelling split."
)

SUMMARY_OUTPUT.write_text(
    "\n".join(summary),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 11. Print final report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE E COMPLETE")
print("=" * 70)

print("\nBalanced dataset:")
print(f"Rows: {len(balanced_clean)}")
print(class_counts(balanced_clean))

print("\nTraining:")
print(f"Rows: {len(train_clean)}")
print(class_counts(train_clean))

print("\nValidation:")
print(f"Rows: {len(validation_clean)}")
print(class_counts(validation_clean))

print("\nTest:")
print(f"Rows: {len(test_clean)}")
print(class_counts(test_clean))

print("\nSpatial blocks:")
print(f"Train      : {len(train_blocks)}")
print(f"Validation : {len(val_blocks)}")
print(f"Test       : {len(test_blocks)}")

print("\nLeakage:")
print(f"Train/Val : {len(train_val_overlap)}")
print(f"Train/Test: {len(train_test_overlap)}")
print(f"Val/Test  : {len(val_test_overlap)}")

print("\nFiles created:")
print(BALANCED_OUTPUT)
print(TRAIN_OUTPUT)
print(VAL_OUTPUT)
print(TEST_OUTPUT)
print(SUMMARY_OUTPUT)

print("\nPhase E is ready for Phase F.")
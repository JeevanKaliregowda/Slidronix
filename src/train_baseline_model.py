import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landslide_inventory_cleaned.csv"
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

print("=" * 60)
print("LANDSLIDE BASELINE ML MODEL")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# --------------------------------------------------
# CLEAN TARGET LABELS
# --------------------------------------------------

df["Movement Type"] = (
    df["Movement Type"]
    .astype(str)
    .str.strip()
    .str.title()
)

print("\nMovement Type Distribution:")
print(df["Movement Type"].value_counts().head(15))


# --------------------------------------------------
# KEEP MEANINGFUL CLASSES
# --------------------------------------------------

# Keep classes with at least 50 records
class_counts = df["Movement Type"].value_counts()

valid_classes = class_counts[
    class_counts >= 50
].index

df = df[
    df["Movement Type"].isin(valid_classes)
].copy()

print("\nClasses used for training:")
print(df["Movement Type"].value_counts())


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

FEATURES = [
    "Latitude",
    "Longitude",
    "State",
    "Material Involved"
]

TARGET = "Movement Type"

X = df[FEATURES]
y = df[TARGET]


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

numeric_features = [
    "Latitude",
    "Longitude"
]

categorical_features = [
    "State",
    "Material Involved"
]


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed!")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

plt.figure(figsize=(10, 8))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation=45,
    values_format="d"
)

plt.title(
    "Baseline Model Confusion Matrix"
)

plt.tight_layout()

CONFUSION_PATH = (
    OUTPUT_DIR
    / "baseline_model_confusion_matrix.png"
)

plt.savefig(
    CONFUSION_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_model_results.txt"
)

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "LANDSLIDE BASELINE ML MODEL RESULTS\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


print("\nConfusion matrix saved to:")
print(CONFUSION_PATH)

print("\nResults saved to:")
print(RESULTS_PATH)

print("\nBASELINE MODEL IMPLEMENTATION COMPLETED!")

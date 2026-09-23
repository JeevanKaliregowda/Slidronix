import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import joblib


# ============================================================
# CONFIGURATION
# ============================================================

PHASE_E_DIR = "data/processed/phase_e"
OUTPUT_DIR = "outputs/phase_g"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Elevation_m",
    "Slope_deg",

    "rainfall_total_30d_mm",
    "rainfall_mean_daily_mm",
    "rainfall_max_daily_mm",
    "rainfall_std_daily_mm",
    "rainfall_7d_mm",
    "rainfall_15d_mm",
    "rainfall_30d_mm",
    "rainfall_max_7d_mm",
    "rainfall_max_15d_mm",
    "rainfall_max_30d_mm",

    "NDVI_mean",
    "NDVI_std",
    "NDVI_min",
    "NDVI_max",
    "NDVI_valid_observations",
    "NDVI_range",

    "Land_Cover_Type",
    "Land_Cover_QC",

    "Soil_clay_0_5cm",
    "Soil_sand_0_5cm",
    "Soil_silt_0_5cm",
    "Soil_soc_0_5cm",
    "Soil_bdod_0_5cm",
    "Soil_phh2o_0_5cm"
]


# ============================================================
# LOAD PHASE E SPLITS
# ============================================================

train_df = pd.read_csv(
    os.path.join(
        PHASE_E_DIR,
        "slidronix_phase_e_train.csv"
    )
)

val_df = pd.read_csv(
    os.path.join(
        PHASE_E_DIR,
        "slidronix_phase_e_validation.csv"
    )
)

test_df = pd.read_csv(
    os.path.join(
        PHASE_E_DIR,
        "slidronix_phase_e_test.csv"
    )
)


print("==================================================")
print("SLIDRONIX PHASE G - RANDOM FOREST")
print("==================================================")

print(
    f"Training samples: {len(train_df)}"
)

print(
    f"Validation samples: {len(val_df)}"
)

print(
    f"Test samples: {len(test_df)}"
)

print(
    f"Features: {len(FEATURES)}"
)


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train_df[
    FEATURES
].copy()

X_val = val_df[
    FEATURES
].copy()

X_test = test_df[
    FEATURES
].copy()


y_train = train_df[
    "Landslide"
].astype(int).values

y_val = val_df[
    "Landslide"
].astype(int).values

y_test = test_df[
    "Landslide"
].astype(int).values


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_val = imputer.transform(
    X_val
)

X_test = imputer.transform(
    X_test
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=25,

    min_samples_leaf=2,

    max_features="sqrt",

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1
)


print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

val_probs = model.predict_proba(
    X_val
)[:, 1]

val_predictions = (
    val_probs >= 0.5
).astype(int)


val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision = precision_score(
    y_val,
    val_predictions,
    zero_division=0
)

val_recall = recall_score(
    y_val,
    val_predictions,
    zero_division=0
)

val_f1 = f1_score(
    y_val,
    val_predictions,
    zero_division=0
)

val_auc = roc_auc_score(
    y_val,
    val_probs
)


print("\nVALIDATION RESULTS")
print(
    f"Accuracy : {val_accuracy:.4f}"
)

print(
    f"Precision: {val_precision:.4f}"
)

print(
    f"Recall   : {val_recall:.4f}"
)

print(
    f"F1       : {val_f1:.4f}"
)

print(
    f"ROC-AUC  : {val_auc:.4f}"
)


# ============================================================
# TEST
# ============================================================

test_probs = model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probs >= 0.5
).astype(int)


accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    test_probs
)

cm = confusion_matrix(
    y_test,
    test_predictions
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": model.feature_importances_

}).sort_values(
    "Importance",
    ascending=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "rf_phase_g_best.joblib"
)

joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "features": FEATURES
    },
    model_path
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "rf_phase_g_results.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SLIDRONIX PHASE G - RANDOM FOREST\n"
    )

    f.write(
        "==================================\n\n"
    )

    f.write(
        f"Training samples: {len(train_df)}\n"
    )

    f.write(
        f"Validation samples: {len(val_df)}\n"
    )

    f.write(
        f"Test samples: {len(test_df)}\n"
    )

    f.write(
        f"Number of features: {len(FEATURES)}\n\n"
    )

    f.write(
        "VALIDATION RESULTS\n"
    )

    f.write(
        "-------------------\n"
    )

    f.write(
        f"Accuracy : {val_accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {val_precision:.4f}\n"
    )

    f.write(
        f"Recall   : {val_recall:.4f}\n"
    )

    f.write(
        f"F1       : {val_f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC  : {val_auc:.4f}\n\n"
    )

    f.write(
        "TEST RESULTS\n"
    )

    f.write(
        "------------\n"
    )

    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall   : {recall:.4f}\n"
    )

    f.write(
        f"F1       : {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC  : {auc:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\nFEATURE IMPORTANCE\n"
    )

    f.write(
        "------------------\n"
    )

    f.write(
        importance.to_string(
            index=False
        )
    )


# ============================================================
# SAVE FEATURE IMPORTANCE CSV
# ============================================================

importance_file = os.path.join(
    OUTPUT_DIR,
    "rf_feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==================================================")
print("RANDOM FOREST TEST RESULTS")
print("==================================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1       : {f1:.4f}"
)

print(
    f"ROC-AUC  : {auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nTop 10 Features:")

print(
    importance.head(10).to_string(
        index=False
    )
)

print(
    f"\nSaved model: {model_path}"
)

print(
    f"Saved results: {results_file}"
)

print(
    f"Saved feature importance: "
    f"{importance_file}"
)
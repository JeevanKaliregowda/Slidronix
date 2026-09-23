import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"

PHASE_E_DIR = os.path.join(
    DATA_DIR,
    "phase_e"
)

CNN_DIR = os.path.join(
    DATA_DIR,
    "phase_f",
    "cnn_spatial"
)

RAINFALL_FILE = os.path.join(
    DATA_DIR,
    "rainfall",
    "chirps_2022_daily_14270.csv"
)

MODEL_FILE = os.path.join(
    "outputs",
    "phase_f",
    "fusion_best.pt"
)

OUTPUT_DIR = os.path.join(
    "outputs",
    "phase_h"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = os.path.join(
    PHASE_E_DIR,
    "slidronix_phase_e_train.csv"
)

TEST_FILE = os.path.join(
    PHASE_E_DIR,
    "slidronix_phase_e_test.csv"
)

CNN_TRAIN_FILE = os.path.join(
    CNN_DIR,
    "train_cnn_patches.npz"
)

CNN_TEST_FILE = os.path.join(
    CNN_DIR,
    "test_cnn_patches.npz"
)


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
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# NORMALIZE IDS
# ============================================================

def normalize_id(series):

    return (
        series
        .astype(str)
        .str.replace(
            "_",
            "",
            regex=False
        )
        .str.strip()
    )


# ============================================================
# EXACT FUSION MODEL
# Copied from train_fusion.py
# ============================================================

class FusionModel(nn.Module):

    def __init__(
        self,
        tabular_features
    ):

        super().__init__()


        # ----------------------------------------------------
        # TABULAR / MLP BRANCH
        # ----------------------------------------------------

        self.tabular_encoder = nn.Sequential(

            nn.Linear(
                tabular_features,
                128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                32
            )
        )


        # ----------------------------------------------------
        # CNN SPATIAL BRANCH
        # ----------------------------------------------------

        self.cnn_encoder = nn.Sequential(

            nn.Conv2d(
                2,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        self.cnn_projection = nn.Sequential(

            nn.Linear(
                128,
                32
            ),

            nn.ReLU()
        )


        # ----------------------------------------------------
        # TRANSFORMER RAINFALL BRANCH
        # ----------------------------------------------------

        d_model = 64

        self.rain_input = nn.Linear(
            1,
            d_model
        )

        self.rain_position = nn.Parameter(
            torch.zeros(
                1,
                30,
                d_model
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=128,
                dropout=0.2,
                batch_first=True,
                activation="gelu"
            )
        )

        self.rain_transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2
            )
        )

        self.rain_projection = nn.Sequential(

            nn.Linear(
                d_model,
                32
            ),

            nn.ReLU()
        )


        # ----------------------------------------------------
        # FUSION HEAD
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                96,
                64
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                32,
                1
            )
        )


    def forward(
        self,
        tabular,
        cnn,
        rainfall
    ):

        # ----------------------------------------------------
        # TABULAR
        # ----------------------------------------------------

        tab_embedding = (
            self.tabular_encoder(
                tabular
            )
        )


        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        cnn_features = (
            self.cnn_encoder(
                cnn
            )
        )

        cnn_features = (
            cnn_features
            .flatten(1)
        )

        cnn_embedding = (
            self.cnn_projection(
                cnn_features
            )
        )


        # ----------------------------------------------------
        # RAINFALL TRANSFORMER
        # ----------------------------------------------------

        rainfall = self.rain_input(
            rainfall
        )

        rainfall = (
            rainfall
            + self.rain_position
        )

        rainfall = (
            self.rain_transformer(
                rainfall
            )
        )

        rainfall = rainfall.mean(
            dim=1
        )

        rain_embedding = (
            self.rain_projection(
                rainfall
            )
        )


        # ----------------------------------------------------
        # FEATURE FUSION
        # ----------------------------------------------------

        fused = torch.cat(
            [
                tab_embedding,
                cnn_embedding,
                rain_embedding
            ],
            dim=1
        )

        logits = self.fusion(
            fused
        )

        return logits.squeeze(1)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "=================================================="
)

print(
    "SLIDRONIX PHASE H - RISK PREDICTION"
)

print(
    "=================================================="
)

print(
    f"\nDevice: {DEVICE}"
)

print(
    "\nLoading Phase E data..."
)


train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


print(
    f"Training rows: {len(train_df)}"
)

print(
    f"Test rows: {len(test_df)}"
)


# ============================================================
# NORMALIZE PHASE E IDS
# ============================================================

train_df["ID_norm"] = normalize_id(
    train_df["ID"]
)

test_df["ID_norm"] = normalize_id(
    test_df["ID"]
)


# ============================================================
# LOAD CNN DATA
# ============================================================

print(
    "\nLoading CNN spatial data..."
)


cnn_train = np.load(
    CNN_TRAIN_FILE,
    allow_pickle=True
)

cnn_test = np.load(
    CNN_TEST_FILE,
    allow_pickle=True
)


cnn_train_X = cnn_train["X"]

cnn_train_ids = np.array(
    [
        str(x)
        .replace("_", "")
        .strip()
        for x in cnn_train["ids"]
    ]
)

cnn_test_X = cnn_test["X"]

cnn_test_ids = np.array(
    [
        str(x)
        .replace("_", "")
        .strip()
        for x in cnn_test["ids"]
    ]
)


print(
    f"CNN train samples: "
    f"{len(cnn_train_X)}"
)

print(
    f"CNN test samples: "
    f"{len(cnn_test_X)}"
)


# ============================================================
# CNN LOOKUPS
# ============================================================

cnn_train_lookup = {
    sample_id: index
    for index, sample_id
    in enumerate(cnn_train_ids)
}

cnn_test_lookup = {
    sample_id: index
    for index, sample_id
    in enumerate(cnn_test_ids)
}


# ============================================================
# LOAD RAINFALL
# ============================================================

print(
    "\nLoading rainfall sequences..."
)


rain_df = pd.read_csv(
    RAINFALL_FILE
)

rain_df["ID_norm"] = normalize_id(
    rain_df["ID"]
)


NON_DAILY_COLUMNS = {
    "ID",
    "Latitude",
    "Longitude",
    "ID_norm"
}


DAILY_COLUMNS = [
    col
    for col in rain_df.columns
    if col not in NON_DAILY_COLUMNS
]


DAILY_COLUMNS = DAILY_COLUMNS[:30]


if len(DAILY_COLUMNS) != 30:

    raise ValueError(
        f"Expected 30 rainfall columns, "
        f"found {len(DAILY_COLUMNS)}"
    )


print(
    f"Rainfall columns: "
    f"{len(DAILY_COLUMNS)}"
)

print(
    DAILY_COLUMNS
)


rain_lookup = rain_df.set_index(
    "ID_norm"
)


# ============================================================
# ALIGN TEST DATA
# ============================================================

print(
    "\nAligning Phase E, CNN and rainfall data..."
)


valid_rows = []

cnn_indices = []

rainfall_sequences = []


for i, row in test_df.iterrows():

    sample_id = row["ID_norm"]


    if sample_id not in cnn_test_lookup:

        continue


    if sample_id not in rain_lookup.index:

        continue


    valid_rows.append(i)


    cnn_indices.append(
        cnn_test_lookup[
            sample_id
        ]
    )


    rainfall_sequences.append(
        rain_lookup.loc[
            sample_id,
            DAILY_COLUMNS
        ].values.astype(
            np.float32
        )
    )


aligned_test_df = (
    test_df
    .loc[valid_rows]
    .copy()
    .reset_index(drop=True)
)


print(
    f"Aligned test samples: "
    f"{len(aligned_test_df)}"
)


if len(aligned_test_df) == 0:

    raise ValueError(
        "No aligned test samples found."
    )


# ============================================================
# BUILD TEST TABULAR DATA
# ============================================================

X_train_tab = train_df[
    FEATURES
].copy()

X_test_tab = aligned_test_df[
    FEATURES
].copy()


# ============================================================
# EXACT TABULAR PREPROCESSING
# FIT ONLY ON TRAIN
# ============================================================

print(
    "\nPreprocessing tabular features..."
)


imputer = SimpleImputer(
    strategy="median"
)

scaler = StandardScaler()


X_train_tab = imputer.fit_transform(
    X_train_tab
)

X_test_tab = imputer.transform(
    X_test_tab
)


X_train_tab = scaler.fit_transform(
    X_train_tab
)

X_test_tab = scaler.transform(
    X_test_tab
)


# ============================================================
# CNN PREPROCESSING
# FIT ONLY ON TRAIN
# ============================================================

print(
    "Preprocessing CNN patches..."
)


X_train_cnn = cnn_train_X.astype(
    np.float32
)

X_test_cnn = cnn_test_X[
    cnn_indices
].astype(
    np.float32
)


cnn_mean = (
    X_train_cnn
    .mean(
        axis=(0, 2, 3),
        keepdims=True
    )
)


cnn_std = (
    X_train_cnn
    .std(
        axis=(0, 2, 3),
        keepdims=True
    )
)


cnn_std[
    cnn_std < 1e-8
] = 1.0


X_test_cnn = (
    X_test_cnn
    - cnn_mean
) / cnn_std


# ============================================================
# RAINFALL PREPROCESSING
# FIT ONLY ON TRAIN
# ============================================================

print(
    "Preprocessing rainfall sequences..."
)


# Get training rainfall sequences
train_rainfall_sequences = []


for sample_id in train_df[
    "ID_norm"
]:

    if sample_id not in rain_lookup.index:

        continue


    train_rainfall_sequences.append(
        rain_lookup.loc[
            sample_id,
            DAILY_COLUMNS
        ].values.astype(
            np.float32
        )
    )


X_train_rain = np.array(
    train_rainfall_sequences,
    dtype=np.float32
)


X_test_rain = np.array(
    rainfall_sequences,
    dtype=np.float32
)


print(
    f"Training rainfall sequences: "
    f"{X_train_rain.shape}"
)

print(
    f"Test rainfall sequences: "
    f"{X_test_rain.shape}"
)


rain_mean = X_train_rain.mean(
    axis=0
)

rain_std = X_train_rain.std(
    axis=0
)


rain_std[
    rain_std < 1e-8
] = 1.0


X_test_rain = (
    X_test_rain
    - rain_mean
) / rain_std


# ============================================================
# CONVERT TO TORCH
# ============================================================

X_test_tab = torch.tensor(
    X_test_tab,
    dtype=torch.float32
)

X_test_cnn = torch.tensor(
    X_test_cnn,
    dtype=torch.float32
)

X_test_rain = torch.tensor(
    X_test_rain,
    dtype=torch.float32
).unsqueeze(-1)


print(
    "\nFinal Phase H tensors:"
)

print(
    f"Tabular: {X_test_tab.shape}"
)

print(
    f"CNN: {X_test_cnn.shape}"
)

print(
    f"Rainfall: {X_test_rain.shape}"
)


# ============================================================
# LOAD EXACT TRAINED FUSION MODEL
# ============================================================

print(
    "\nLoading trained Fusion model..."
)


model = FusionModel(
    tabular_features=len(FEATURES)
).to(
    DEVICE
)


state_dict = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)


model.load_state_dict(
    state_dict
)


model.eval()


print(
    "Fusion model loaded successfully."
)


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

print(
    "\nGenerating landslide probabilities..."
)


with torch.no_grad():

    logits = model(
        X_test_tab.to(DEVICE),

        X_test_cnn.to(DEVICE),

        X_test_rain.to(DEVICE)
    )

    probabilities = (
        torch.sigmoid(
            logits
        )
        .cpu()
        .numpy()
    )


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(
    probability
):

    if probability < 0.33:

        return "Low"

    elif probability < 0.66:

        return "Moderate"

    else:

        return "High"


aligned_test_df[
    "Landslide_Probability"
] = probabilities


aligned_test_df[
    "Risk_Level"
] = [
    classify_risk(
        probability
    )
    for probability
    in probabilities
]


# ============================================================
# CREATE OUTPUT
# ============================================================

OUTPUT_COLUMNS = [

    "ID",

    "Latitude",

    "Longitude",

    "Elevation_m",

    "Slope_deg",

    "rainfall_total_30d_mm",

    "rainfall_mean_daily_mm",

    "rainfall_max_daily_mm",

    "NDVI_mean",

    "Land_Cover_Type",

    "Soil_clay_0_5cm",

    "Soil_sand_0_5cm",

    "Soil_silt_0_5cm",

    "Soil_soc_0_5cm",

    "Soil_bdod_0_5cm",

    "Soil_phh2o_0_5cm",

    "Landslide_Probability",

    "Risk_Level"
]


output_df = aligned_test_df[
    OUTPUT_COLUMNS
].copy()


# ============================================================
# SAVE CSV
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "slidronix_phase_h_risk_predictions.csv"
)


output_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# RISK SUMMARY
# ============================================================

risk_counts = (
    output_df[
        "Risk_Level"
    ]
    .value_counts()
)


summary_file = os.path.join(
    OUTPUT_DIR,
    "phase_h_risk_summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SLIDRONIX PHASE H - RISK ASSESSMENT\n"
    )

    f.write(
        "====================================\n\n"
    )

    f.write(
        f"Prediction samples: "
        f"{len(output_df)}\n"
    )

    f.write(
        f"Minimum probability: "
        f"{probabilities.min():.4f}\n"
    )

    f.write(
        f"Maximum probability: "
        f"{probabilities.max():.4f}\n"
    )

    f.write(
        f"Mean probability: "
        f"{probabilities.mean():.4f}\n\n"
    )

    f.write(
        "Risk distribution:\n"
    )

    f.write(
        risk_counts.to_string()
    )

    f.write(
        "\n\nRisk thresholds:\n"
    )

    f.write(
        "Low      : probability < 0.33\n"
    )

    f.write(
        "Moderate : 0.33 <= probability < 0.66\n"
    )

    f.write(
        "High     : probability >= 0.66\n"
    )

    f.write(
        "\n\nImportant limitation:\n"
    )

    f.write(
        "The rainfall input currently represents "
        "the fixed January 1-30, 2022 reference "
        "period. It is not a real-time rainfall "
        "trigger.\n"
    )

    f.write(
        "\nThe Low/Moderate/High thresholds are "
        "project-defined visualization thresholds "
        "and are not validated operational warning "
        "thresholds.\n"
    )


# ============================================================
# TOP-RISK LOCATIONS
# ============================================================

top_risk = (
    output_df
    .sort_values(
        "Landslide_Probability",
        ascending=False
    )
    .head(10)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n=================================================="
)

print(
    "PHASE H RISK PREDICTION COMPLETE"
)

print(
    "=================================================="
)

print(
    f"\nPrediction samples: "
    f"{len(output_df)}"
)

print(
    f"Probability range: "
    f"{probabilities.min():.4f} - "
    f"{probabilities.max():.4f}"
)

print(
    f"Mean probability: "
    f"{probabilities.mean():.4f}"
)


print(
    "\nRisk distribution:"
)

print(
    risk_counts
)


print(
    "\nTop 10 highest predicted-risk locations:"
)

print(
    top_risk[
        [
            "ID",
            "Latitude",
            "Longitude",
            "Landslide_Probability",
            "Risk_Level"
        ]
    ].to_string(
        index=False
    )
)


print(
    f"\nSaved risk predictions:"
)

print(
    output_file
)


print(
    f"\nSaved risk summary:"
)

print(
    summary_file
)


print(
    "\nPHASE H STEP 1 COMPLETE"
)
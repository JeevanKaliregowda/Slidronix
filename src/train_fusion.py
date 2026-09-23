import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

DATA_DIR = "data/processed"
PHASE_E_DIR = os.path.join(DATA_DIR, "phase_e")
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

OUTPUT_DIR = "outputs/phase_f"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = os.path.join(
    PHASE_E_DIR,
    "slidronix_phase_e_train.csv"
)

VAL_FILE = os.path.join(
    PHASE_E_DIR,
    "slidronix_phase_e_validation.csv"
)

TEST_FILE = os.path.join(
    PHASE_E_DIR,
    "slidronix_phase_e_test.csv"
)

CNN_TRAIN_FILE = os.path.join(
    CNN_DIR,
    "train_cnn_patches.npz"
)

CNN_VAL_FILE = os.path.join(
    CNN_DIR,
    "validation_cnn_patches.npz"
)

CNN_TEST_FILE = os.path.join(
    CNN_DIR,
    "test_cnn_patches.npz"
)


# ============================================================
# MLP FEATURES
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
# LOAD PHASE E DATA
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)

val_df = pd.read_csv(
    VAL_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


print("==================================================")
print("SLIDRONIX PHASE F - MULTIMODAL FUSION")
print("==================================================")

print(
    f"Device: {DEVICE}"
)

print(
    f"Phase E train: {train_df.shape}"
)

print(
    f"Phase E validation: {val_df.shape}"
)

print(
    f"Phase E test: {test_df.shape}"
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


train_df["ID_norm"] = normalize_id(
    train_df["ID"]
)

val_df["ID_norm"] = normalize_id(
    val_df["ID"]
)

test_df["ID_norm"] = normalize_id(
    test_df["ID"]
)


# ============================================================
# LOAD CNN DATA
# ============================================================

print("\nLoading CNN spatial datasets...")

cnn_train = np.load(
    CNN_TRAIN_FILE,
    allow_pickle=True
)

cnn_val = np.load(
    CNN_VAL_FILE,
    allow_pickle=True
)

cnn_test = np.load(
    CNN_TEST_FILE,
    allow_pickle=True
)


cnn_train_X = cnn_train["X"]
cnn_train_y = cnn_train["y"]
cnn_train_ids = cnn_train["ids"]

cnn_val_X = cnn_val["X"]
cnn_val_y = cnn_val["y"]
cnn_val_ids = cnn_val["ids"]

cnn_test_X = cnn_test["X"]
cnn_test_y = cnn_test["y"]
cnn_test_ids = cnn_test["ids"]


print(
    "CNN train:",
    cnn_train_X.shape
)

print(
    "CNN validation:",
    cnn_val_X.shape
)

print(
    "CNN test:",
    cnn_test_X.shape
)


# ============================================================
# NORMALIZE CNN IDS
# ============================================================

cnn_train_ids = np.array(
    [
        str(x).replace("_", "").strip()
        for x in cnn_train_ids
    ]
)

cnn_val_ids = np.array(
    [
        str(x).replace("_", "").strip()
        for x in cnn_val_ids
    ]
)

cnn_test_ids = np.array(
    [
        str(x).replace("_", "").strip()
        for x in cnn_test_ids
    ]
)


# ============================================================
# LOAD RAINFALL DATA
# ============================================================

print(
    "\nLoading rainfall sequences..."
)

rain_df = pd.read_csv(
    RAINFALL_FILE
)

NON_DAILY_COLUMNS = {
    "ID",
    "Latitude",
    "Longitude"
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


rain_df["ID_norm"] = normalize_id(
    rain_df["ID"]
)

rain_lookup = rain_df.set_index(
    "ID_norm"
)


# ============================================================
# CNN LOOKUP INDICES
# ============================================================

cnn_train_lookup = {
    id_: i
    for i, id_ in enumerate(
        cnn_train_ids
    )
}

cnn_val_lookup = {
    id_: i
    for i, id_ in enumerate(
        cnn_val_ids
    )
}

cnn_test_lookup = {
    id_: i
    for i, id_ in enumerate(
        cnn_test_ids
    )
}


# ============================================================
# BUILD ALIGNED MULTIMODAL SPLIT
# ============================================================

def build_split(
    phase_df,
    cnn_X,
    cnn_y,
    cnn_ids,
    cnn_lookup,
    split_name
):

    valid_rows = []

    for i, row in phase_df.iterrows():

        sample_id = row["ID_norm"]

        if sample_id not in cnn_lookup:
            continue

        if sample_id not in rain_lookup.index:
            continue

        valid_rows.append(i)


    valid_df = (
        phase_df
        .loc[valid_rows]
        .copy()
    )

    # --------------------------------------------------------
    # Tabular features
    # --------------------------------------------------------

    X_tabular = valid_df[
        FEATURES
    ].copy()

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y = (
        valid_df["Landslide"]
        .values
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # CNN patches
    # --------------------------------------------------------

    cnn_indices = [
        cnn_lookup[id_]
        for id_ in valid_df["ID_norm"]
    ]

    X_cnn = cnn_X[
        cnn_indices
    ].astype(np.float32)

    # --------------------------------------------------------
    # Rainfall sequences
    # --------------------------------------------------------

    X_rain = rain_lookup.loc[
        valid_df["ID_norm"],
        DAILY_COLUMNS
    ].values.astype(
        np.float32
    )

    ids = (
        valid_df["ID"]
        .values
    )

    print(
        f"\n{split_name} aligned samples: "
        f"{len(valid_df)}"
    )

    print(
        f"{split_name} CNN shape: "
        f"{X_cnn.shape}"
    )

    print(
        f"{split_name} rainfall shape: "
        f"{X_rain.shape}"
    )

    print(
        f"{split_name} tabular shape: "
        f"{X_tabular.shape}"
    )

    return (
        X_tabular,
        X_cnn,
        X_rain,
        y,
        ids
    )


(
    X_train_tab,
    X_train_cnn,
    X_train_rain,
    y_train,
    train_ids
) = build_split(
    train_df,
    cnn_train_X,
    cnn_train_y,
    cnn_train_ids,
    cnn_train_lookup,
    "Train"
)


(
    X_val_tab,
    X_val_cnn,
    X_val_rain,
    y_val,
    val_ids
) = build_split(
    val_df,
    cnn_val_X,
    cnn_val_y,
    cnn_val_ids,
    cnn_val_lookup,
    "Validation"
)


(
    X_test_tab,
    X_test_cnn,
    X_test_rain,
    y_test,
    test_ids
) = build_split(
    test_df,
    cnn_test_X,
    cnn_test_y,
    cnn_test_ids,
    cnn_test_lookup,
    "Test"
)


# ============================================================
# VERIFY LABEL AGREEMENT
# ============================================================

def verify_labels(
    phase_df,
    ids,
    cnn_y,
    cnn_lookup,
    split_name
):

    phase_lookup = {
        str(row["ID"]).replace("_", ""):
        int(row["Landslide"])
        for _, row in phase_df.iterrows()
    }

    mismatches = 0

    for id_ in ids:

        norm_id = (
            str(id_)
            .replace("_", "")
        )

        cnn_index = cnn_lookup[
            norm_id
        ]

        cnn_label = int(
            cnn_y[cnn_index]
        )

        phase_label = (
            phase_lookup[norm_id]
        )

        if cnn_label != phase_label:
            mismatches += 1

    print(
        f"{split_name} label mismatches: "
        f"{mismatches}"
    )

    if mismatches > 0:

        raise ValueError(
            f"Label mismatch detected in "
            f"{split_name} data."
        )


verify_labels(
    train_df,
    train_ids,
    cnn_train_y,
    cnn_train_lookup,
    "Train"
)

verify_labels(
    val_df,
    val_ids,
    cnn_val_y,
    cnn_val_lookup,
    "Validation"
)

verify_labels(
    test_df,
    test_ids,
    cnn_test_y,
    cnn_test_lookup,
    "Test"
)


# ============================================================
# PREPROCESS TABULAR FEATURES
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

scaler = StandardScaler()


X_train_tab = imputer.fit_transform(
    X_train_tab
)

X_val_tab = imputer.transform(
    X_val_tab
)

X_test_tab = imputer.transform(
    X_test_tab
)


X_train_tab = scaler.fit_transform(
    X_train_tab
)

X_val_tab = scaler.transform(
    X_val_tab
)

X_test_tab = scaler.transform(
    X_test_tab
)


# ============================================================
# PREPROCESS CNN PATCHES
# ============================================================

# Compute channel statistics using
# training data only.

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

cnn_std[cnn_std < 1e-8] = 1.0


X_train_cnn = (
    X_train_cnn -
    cnn_mean
) / cnn_std

X_val_cnn = (
    X_val_cnn -
    cnn_mean
) / cnn_std

X_test_cnn = (
    X_test_cnn -
    cnn_mean
) / cnn_std


# ============================================================
# PREPROCESS RAINFALL
# ============================================================

rain_mean = X_train_rain.mean(
    axis=0
)

rain_std = X_train_rain.std(
    axis=0
)

rain_std[
    rain_std < 1e-8
] = 1.0


X_train_rain = (
    X_train_rain -
    rain_mean
) / rain_std

X_val_rain = (
    X_val_rain -
    rain_mean
) / rain_std

X_test_rain = (
    X_test_rain -
    rain_mean
) / rain_std


# ============================================================
# CONVERT TO TORCH TENSORS
# ============================================================

X_train_tab = torch.tensor(
    X_train_tab,
    dtype=torch.float32
)

X_val_tab = torch.tensor(
    X_val_tab,
    dtype=torch.float32
)

X_test_tab = torch.tensor(
    X_test_tab,
    dtype=torch.float32
)


X_train_cnn = torch.tensor(
    X_train_cnn,
    dtype=torch.float32
)

X_val_cnn = torch.tensor(
    X_val_cnn,
    dtype=torch.float32
)

X_test_cnn = torch.tensor(
    X_test_cnn,
    dtype=torch.float32
)


X_train_rain = torch.tensor(
    X_train_rain,
    dtype=torch.float32
).unsqueeze(-1)

X_val_rain = torch.tensor(
    X_val_rain,
    dtype=torch.float32
).unsqueeze(-1)

X_test_rain = torch.tensor(
    X_test_rain,
    dtype=torch.float32
).unsqueeze(-1)


y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
)


print("\nFinal tensor shapes:")

print(
    "Train tabular:",
    X_train_tab.shape
)

print(
    "Train CNN:",
    X_train_cnn.shape
)

print(
    "Train rainfall:",
    X_train_rain.shape
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    TensorDataset(
        X_train_tab,
        X_train_cnn,
        X_train_rain,
        y_train
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    TensorDataset(
        X_val_tab,
        X_val_cnn,
        X_val_rain,
        y_val
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(
        X_test_tab,
        X_test_cnn,
        X_test_rain,
        y_test
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# FUSION MODEL
# ============================================================

class FusionModel(
    nn.Module
):

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

        # 32 tabular
        # + 32 CNN
        # + 32 rainfall
        #
        # = 96 dimensional fused representation

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
        # TABULAR EMBEDDING
        # ----------------------------------------------------

        tab_embedding = (
            self.tabular_encoder(
                tabular
            )
        )


        # ----------------------------------------------------
        # CNN EMBEDDING
        # ----------------------------------------------------

        cnn_features = (
            self.cnn_encoder(
                cnn
            )
        )

        cnn_features = (
            cnn_features
            .flatten(
                1
            )
        )

        cnn_embedding = (
            self.cnn_projection(
                cnn_features
            )
        )


        # ----------------------------------------------------
        # RAINFALL EMBEDDING
        # ----------------------------------------------------

        rainfall = self.rain_input(
            rainfall
        )

        rainfall = (
            rainfall +
            self.rain_position
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
# CREATE MODEL
# ============================================================

model = FusionModel(
    tabular_features=len(FEATURES)
).to(
    DEVICE
)


print("\nFusion model:")
print(model)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for (
            tabular,
            cnn,
            rainfall,
            labels
        ) in loader:

            tabular = tabular.to(
                DEVICE
            )

            cnn = cnn.to(
                DEVICE
            )

            rainfall = rainfall.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            logits = model(
                tabular,
                cnn,
                rainfall
            )

            probs = torch.sigmoid(
                logits
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


    probs = np.array(
        all_probs
    )

    labels = np.array(
        all_labels
    )

    predictions = (
        probs >= 0.5
    ).astype(int)


    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        labels,
        probs
    )


    return (
        accuracy,
        precision,
        recall,
        f1,
        auc,
        probs,
        predictions,
        labels
    )


# ============================================================
# TRAIN
# ============================================================

best_val_auc = -1.0
best_epoch = 0

model_path = os.path.join(
    OUTPUT_DIR,
    "fusion_best.pt"
)


print(
    "\n=================================================="
)

print(
    "TRAINING MULTIMODAL FUSION MODEL"
)

print(
    "=================================================="
)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_loss = 0.0


    for (
        tabular,
        cnn,
        rainfall,
        labels
    ) in train_loader:

        tabular = tabular.to(
            DEVICE
        )

        cnn = cnn.to(
            DEVICE
        )

        rainfall = rainfall.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        logits = model(
            tabular,
            cnn,
            rainfall
        )


        loss = criterion(
            logits,
            labels
        )


        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
        )


    avg_loss = (
        total_loss /
        len(train_loader)
    )


    (
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        val_auc,
        _,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )


    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )


    if val_auc > best_val_auc:

        best_val_auc = val_auc

        best_epoch = epoch

        torch.save(
            model.state_dict(),
            model_path
        )


# ============================================================
# TEST
# ============================================================

model.load_state_dict(
    torch.load(
        model_path,
        map_location=DEVICE
    )
)


(
    accuracy,
    precision,
    recall,
    f1,
    auc,
    probs,
    predictions,
    labels
) = evaluate(
    model,
    test_loader
)


cm = confusion_matrix(
    labels,
    predictions
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "fusion_results.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SLIDRONIX PHASE F - "
        "MULTIMODAL FUSION MODEL\n"
    )

    f.write(
        "==========================================\n\n"
    )

    f.write(
        f"Device: {DEVICE}\n"
    )

    f.write(
        f"Training samples: "
        f"{len(X_train_tab)}\n"
    )

    f.write(
        f"Validation samples: "
        f"{len(X_val_tab)}\n"
    )

    f.write(
        f"Test samples: "
        f"{len(X_test_tab)}\n"
    )

    f.write(
        f"Tabular features: "
        f"{len(FEATURES)}\n"
    )

    f.write(
        "CNN input: 2 x 32 x 32\n"
    )

    f.write(
        "Rainfall input: 30 x 1\n"
    )

    f.write(
        f"Best validation epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Best validation ROC-AUC: "
        f"{best_val_auc:.4f}\n\n"
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


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n=================================================="
)

print(
    "FUSION TEST RESULTS"
)

print(
    "=================================================="
)

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

print(
    "\nConfusion Matrix:"
)

print(cm)

print(
    f"\nBest validation epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation AUC: "
    f"{best_val_auc:.4f}"
)

print(
    f"\nSaved model: "
    f"{model_path}"
)

print(
    f"Saved results: "
    f"{results_file}"
)
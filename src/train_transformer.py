import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
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
EPOCHS = 12
LEARNING_RATE = 0.001

DATA_DIR = "data/processed"
PHASE_E_DIR = os.path.join(DATA_DIR, "phase_e")

RAINFALL_FILE = os.path.join(
    DATA_DIR,
    "rainfall",
    "chirps_2022_daily_14270.csv"
)

OUTPUT_DIR = "outputs/phase_f"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD DATA
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

rain_df = pd.read_csv(
    RAINFALL_FILE
)


print("==================================================")
print("SLIDRONIX PHASE F - RAINFALL TRANSFORMER")
print("==================================================")

print(f"Device: {DEVICE}")

print(f"Phase E train: {train_df.shape}")
print(f"Phase E validation: {val_df.shape}")
print(f"Phase E test: {test_df.shape}")
print(f"Rainfall dataset: {rain_df.shape}")


# ============================================================
# IDENTIFY 30 DAILY RAINFALL FEATURES
# ============================================================

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
        f"Expected 30 daily rainfall columns, "
        f"found {len(DAILY_COLUMNS)}"
    )


print("\nDaily rainfall features:")

for i, col in enumerate(
    DAILY_COLUMNS,
    start=1
):
    print(
        f"Day {i:02d}: {col}"
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

rain_df["ID_norm"] = normalize_id(
    rain_df["ID"]
)


# ============================================================
# CREATE RAINFALL LOOKUP
# ============================================================

rain_lookup = rain_df.set_index(
    "ID_norm"
)


# ============================================================
# CHECK ID MATCHING
# ============================================================

def check_ids(
    split_df,
    split_name
):

    missing = (
        ~split_df["ID_norm"]
        .isin(rain_lookup.index)
    )

    missing_count = int(
        missing.sum()
    )

    matched_count = (
        len(split_df) -
        missing_count
    )

    print(
        f"{split_name} rainfall ID matches: "
        f"{matched_count}/{len(split_df)}"
    )

    if missing_count > 0:

        print(
            f"WARNING: {missing_count} IDs "
            f"missing rainfall data."
        )

    return missing_count


check_ids(
    train_df,
    "Train"
)

check_ids(
    val_df,
    "Validation"
)

check_ids(
    test_df,
    "Test"
)


# ============================================================
# BUILD DATA SPLITS
# ============================================================

def build_split(
    split_df,
    split_name
):

    valid_mask = (
        split_df["ID_norm"]
        .isin(rain_lookup.index)
    )

    split_valid = (
        split_df
        .loc[valid_mask]
        .copy()
    )

    rainfall = (
        rain_lookup.loc[
            split_valid["ID_norm"],
            DAILY_COLUMNS
        ]
        .values
        .astype(np.float32)
    )

    labels = (
        split_valid["Landslide"]
        .values
        .astype(np.float32)
    )

    ids = (
        split_valid["ID"]
        .values
    )

    print(
        f"{split_name}: "
        f"{len(rainfall)} samples, "
        f"sequence shape = "
        f"{rainfall.shape}"
    )

    return (
        rainfall,
        labels,
        ids
    )


X_train, y_train, train_ids = build_split(
    train_df,
    "Train"
)

X_val, y_val, val_ids = build_split(
    val_df,
    "Validation"
)

X_test, y_test, test_ids = build_split(
    test_df,
    "Test"
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

train_feature_means = np.nanmean(
    X_train,
    axis=0
)


def fill_missing(X):

    X = X.copy()

    for i in range(
        X.shape[1]
    ):

        mask = np.isnan(
            X[:, i]
        )

        if np.any(mask):

            X[mask, i] = (
                train_feature_means[i]
            )

    return X


X_train = fill_missing(
    X_train
)

X_val = fill_missing(
    X_val
)

X_test = fill_missing(
    X_test
)


# ============================================================
# STANDARDIZATION
# ============================================================

mean = X_train.mean(
    axis=0
)

std = X_train.std(
    axis=0
)

std[std < 1e-8] = 1.0

X_train = (
    X_train - mean
) / std

X_val = (
    X_val - mean
) / std

X_test = (
    X_test - mean
) / std


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

# Transformer input:
#
# [batch, sequence_length, features]
#
# sequence_length = 30 days
# features = 1 rainfall value


X_train = torch.tensor(
    X_train,
    dtype=torch.float32
).unsqueeze(-1)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
).unsqueeze(-1)

X_test = torch.tensor(
    X_test,
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


print("\nTensor shapes:")

print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    TensorDataset(
        X_train,
        y_train
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    TensorDataset(
        X_val,
        y_val
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(
        X_test,
        y_test
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class RainfallTransformer(
    nn.Module
):

    def __init__(
        self,
        input_dim=1,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2
    ):

        super().__init__()

        # Convert each rainfall value
        # into a learned embedding.

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        # Learnable position information
        # for the 30 rainfall days.

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                30,
                d_model
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu"
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                d_model,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                64,
                1
            )
        )


    def forward(self, x):

        # x shape:
        # [batch, 30, 1]

        x = self.input_projection(
            x
        )

        x = (
            x +
            self.position_embedding
        )

        x = self.transformer(
            x
        )

        # Global average pooling
        # across the 30 rainfall days.

        x = x.mean(
            dim=1
        )

        x = self.dropout(
            x
        )

        logits = self.classifier(
            x
        )

        return logits.squeeze(1)


model = RainfallTransformer().to(
    DEVICE
)


print("\nModel:")
print(model)


# ============================================================
# LOSS AND OPTIMIZER
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

        for X, y in loader:

            X = X.to(
                DEVICE
            )

            y = y.to(
                DEVICE
            )

            logits = model(
                X
            )

            probs = torch.sigmoid(
                logits
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
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
# TRAINING
# ============================================================

best_val_auc = -1.0
best_epoch = 0

model_path = os.path.join(
    OUTPUT_DIR,
    "transformer_best.pt"
)


print(
    "\n=================================================="
)

print(
    "TRAINING TRANSFORMER"
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


    for X, y in train_loader:

        X = X.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            X
        )

        loss = criterion(
            logits,
            y
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
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        model_path,
        map_location=DEVICE
    )
)


# ============================================================
# TEST RESULTS
# ============================================================

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
    "transformer_results.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SLIDRONIX PHASE F - "
        "RAINFALL TRANSFORMER\n"
    )

    f.write(
        "==========================================\n\n"
    )

    f.write(
        f"Device: {DEVICE}\n"
    )

    f.write(
        f"Training samples: "
        f"{len(X_train)}\n"
    )

    f.write(
        f"Validation samples: "
        f"{len(X_val)}\n"
    )

    f.write(
        f"Test samples: "
        f"{len(X_test)}\n"
    )

    f.write(
        "Sequence length: 30 days\n"
    )

    f.write(
        "Input features per timestep: "
        "1\n"
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
# PRINT FINAL RESULTS
# ============================================================

print(
    "\n=================================================="
)

print(
    "TRANSFORMER TEST RESULTS"
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
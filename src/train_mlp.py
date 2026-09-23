import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

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


DATA_DIR = "data/processed/phase_e"
OUTPUT_DIR = "outputs/phase_f"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

BATCH_SIZE = 64
EPOCHS = 12
LR = 0.001


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


def load_split(name):

    path = os.path.join(
        DATA_DIR,
        f"slidronix_phase_e_{name}.csv"
    )

    df = pd.read_csv(path)

    X = df[FEATURES].copy()

    y = df["Landslide"].astype(int).values

    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")
X_test, y_test = load_split("test")


# Missing-value handling
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)


# Standardization
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

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


train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE
)


class LandslideMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(len(FEATURES), 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )


    def forward(self, x):

        return self.network(x).squeeze(1)


model = LandslideMLP().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


def evaluate(loader):

    model.eval()

    predictions = []
    probabilities = []
    targets = []

    with torch.no_grad():

        for X, y in loader:

            logits = model(X)

            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.numpy()
            )

            predictions.extend(
                (probs >= 0.5)
                .numpy()
                .astype(int)
            )

            targets.extend(
                y.numpy().astype(int)
            )

    return (
        accuracy_score(targets, predictions),
        precision_score(
            targets,
            predictions,
            zero_division=0
        ),
        recall_score(
            targets,
            predictions,
            zero_division=0
        ),
        f1_score(
            targets,
            predictions,
            zero_division=0
        ),
        roc_auc_score(
            targets,
            probabilities
        ),
        confusion_matrix(
            targets,
            predictions
        )
    )


best_auc = 0.0


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for X, y in train_loader:

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    (
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        val_auc,
        _
    ) = evaluate(val_loader)


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )


    if val_auc > best_auc:

        best_auc = val_auc

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "imputer": imputer,

                "scaler": scaler
            },
            os.path.join(
                OUTPUT_DIR,
                "mlp_best.pt"
            )
        )


checkpoint = torch.load(
    os.path.join(
        OUTPUT_DIR,
        "mlp_best.pt"
    ),
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


(
    test_acc,
    test_precision,
    test_recall,
    test_f1,
    test_auc,
    cm
) = evaluate(test_loader)


print("\n" + "=" * 50)
print("MLP TEST RESULTS")
print("=" * 50)

print("Accuracy :", round(test_acc, 4))
print("Precision:", round(test_precision, 4))
print("Recall   :", round(test_recall, 4))
print("F1       :", round(test_f1, 4))
print("ROC-AUC  :", round(test_auc, 4))

print("\nConfusion Matrix:")
print(cm)


with open(
    os.path.join(
        OUTPUT_DIR,
        "mlp_results.txt"
    ),
    "w"
) as f:

    f.write("SLIDRONIX MLP RESULTS\n")
    f.write("=====================\n\n")

    f.write(f"Accuracy: {test_acc:.4f}\n")
    f.write(f"Precision: {test_precision:.4f}\n")
    f.write(f"Recall: {test_recall:.4f}\n")
    f.write(f"F1: {test_f1:.4f}\n")
    f.write(f"ROC-AUC: {test_auc:.4f}\n")

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))


print(
    "\nSaved:",
    os.path.join(
        OUTPUT_DIR,
        "mlp_results.txt"
    )
)
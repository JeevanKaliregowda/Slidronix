import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

DATA_DIR = "data/processed/phase_f/cnn_spatial"
OUTPUT_DIR = "outputs/phase_f"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

BATCH_SIZE = 64
EPOCHS = 8
LR = 0.001


def load_data(name):
    data = np.load(
        os.path.join(
            DATA_DIR,
            f"{name}_cnn_patches.npz"
        )
    )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    return torch.tensor(X), torch.tensor(y)


X_train, y_train = load_data("train")
X_val, y_val = load_data("validation")
X_test, y_test = load_data("test")


# Normalize each channel using training statistics
mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
std = X_train.std(dim=(0, 2, 3), keepdim=True)

std = torch.clamp(std, min=1e-6)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)


class LandslideCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                2, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )


    def forward(self, x):

        x = self.features(x)

        return self.classifier(x).squeeze(1)


model = LandslideCNN().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


def evaluate(loader):

    model.eval()

    predictions = []
    probabilities = []
    targets = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)

            logits = model(X)

            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy()
            )

            predictions.extend(
                (probs >= 0.5)
                .cpu()
                .numpy()
                .astype(int)
            )

            targets.extend(
                y.numpy().astype(int)
            )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    precision = precision_score(
        targets,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        targets,
        probabilities
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        auc,
        confusion_matrix(
            targets,
            predictions
        )
    )


print("Device:", DEVICE)
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

best_val_auc = 0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for X, y in train_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

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

    if val_auc > best_val_auc:

        best_val_auc = val_auc

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "mean":
                    mean,

                "std":
                    std
            },
            os.path.join(
                OUTPUT_DIR,
                "cnn_best.pt"
            )
        )


# Load best model
checkpoint = torch.load(
    os.path.join(
        OUTPUT_DIR,
        "cnn_best.pt"
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
print("CNN TEST RESULTS")
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
        "cnn_results.txt"
    ),
    "w"
) as f:

    f.write("SLIDRONIX CNN RESULTS\n")
    f.write("=====================\n\n")

    f.write(
        f"Accuracy: {test_acc:.4f}\n"
    )

    f.write(
        f"Precision: {test_precision:.4f}\n"
    )

    f.write(
        f"Recall: {test_recall:.4f}\n"
    )

    f.write(
        f"F1: {test_f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC: {test_auc:.4f}\n"
    )

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

print(
    "\nSaved:",
    os.path.join(
        OUTPUT_DIR,
        "cnn_results.txt"
    )
)
import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs/phase_g"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the actual Phase G comparison results
csv_path = os.path.join(
    OUTPUT_DIR,
    "phase_g_model_comparison.csv"
)

df = pd.read_csv(csv_path)


# ============================================================
# CHART 1 — ALL METRICS
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

plt.figure(figsize=(14, 8))

x = range(len(df))
width = 0.15

for i, metric in enumerate(metrics):

    positions = [
        p + (i - 2) * width
        for p in x
    ]

    plt.bar(
        positions,
        df[metric],
        width=width,
        label=metric
    )

plt.xticks(
    list(x),
    df["Model"],
    fontsize=12
)

plt.ylabel(
    "Score",
    fontsize=13
)

plt.title(
    "Slidronix Phase G — Model Performance Comparison",
    fontsize=18,
    fontweight="bold"
)

plt.ylim(0, 1)

plt.yticks(fontsize=11)

plt.legend(
    fontsize=11,
    ncol=3
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

path1 = os.path.join(
    OUTPUT_DIR,
    "phase_g_ppt_all_metrics.png"
)

plt.savefig(
    path1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CHART 2 — ROC-AUC
# ============================================================

plt.figure(figsize=(12, 7))

bars = plt.bar(
    df["Model"],
    df["ROC-AUC"]
)

plt.ylabel(
    "ROC-AUC",
    fontsize=14
)

plt.xlabel(
    "Model",
    fontsize=14
)

plt.title(
    "Slidronix Phase G — ROC-AUC Comparison",
    fontsize=18,
    fontweight="bold"
)

plt.ylim(
    0,
    1
)

plt.grid(
    axis="y",
    alpha=0.25
)

# Add values above bars
for bar, value in zip(
    bars,
    df["ROC-AUC"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

plt.xticks(
    fontsize=12
)

plt.yticks(
    fontsize=11
)

plt.tight_layout()

path2 = os.path.join(
    OUTPUT_DIR,
    "phase_g_ppt_roc_auc.png"
)

plt.savefig(
    path2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRINT
# ============================================================

print("\nPPT CHARTS CREATED")
print("==================")

print(
    f"All metrics:\n{path1}"
)

print(
    f"\nROC-AUC:\n{path2}"
)
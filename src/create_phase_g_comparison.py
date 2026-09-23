import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "outputs/phase_g"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL RESULTS
# ============================================================

results = [

    {
        "Model": "Random Forest",
        "Accuracy": 0.8154,
        "Precision": 0.8286,
        "Recall": 0.7949,
        "F1": 0.8114,
        "ROC-AUC": 0.9093,
        "Test Samples": 2800
    },

    {
        "Model": "MLP",
        "Accuracy": 0.8236,
        "Precision": 0.7891,
        "Recall": 0.8828,
        "F1": 0.8333,
        "ROC-AUC": 0.8968,
        "Test Samples": 2800
    },

    {
        "Model": "CNN",
        "Accuracy": 0.8124,
        "Precision": 0.7747,
        "Recall": 0.8831,
        "F1": 0.8254,
        "ROC-AUC": 0.8815,
        "Test Samples": 2761
    },

    {
        "Model": "Transformer",
        "Accuracy": 0.6854,
        "Precision": 0.7144,
        "Recall": 0.6169,
        "F1": 0.6621,
        "ROC-AUC": 0.7322,
        "Test Samples": 2800
    },

    {
        "Model": "Fusion",
        "Accuracy": 0.8218,
        "Precision": 0.8033,
        "Recall": 0.8543,
        "F1": 0.8280,
        "ROC-AUC": 0.9122,
        "Test Samples": 2761
    }
]


df = pd.DataFrame(results)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "phase_g_model_comparison.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

txt_path = os.path.join(
    OUTPUT_DIR,
    "phase_g_model_comparison.txt"
)

with open(
    txt_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SLIDRONIX PHASE G - MODEL COMPARISON\n"
    )

    f.write(
        "====================================\n\n"
    )

    f.write(
        "Models evaluated:\n"
    )

    f.write(
        "1. Random Forest\n"
    )

    f.write(
        "2. MLP\n"
    )

    f.write(
        "3. CNN\n"
    )

    f.write(
        "4. Rainfall Transformer\n"
    )

    f.write(
        "5. Multimodal Fusion\n\n"
    )

    f.write(
        df.to_string(index=False)
    )

    f.write(
        "\n\nTEST-SET COVERAGE NOTE\n"
    )

    f.write(
        "----------------------\n"
    )

    f.write(
        "Random Forest, MLP and Transformer were "
        "evaluated on 2,800 Phase E test samples.\n"
    )

    f.write(
        "CNN and Fusion were evaluated on 2,761 "
        "samples because 39 test locations did not "
        "produce valid 32x32 DEM patches.\n"
    )

    f.write(
        "Therefore, metrics should be interpreted with "
        "this difference in test-set coverage in mind.\n"
    )

    f.write(
        "\nMODEL ARCHITECTURE SUMMARY\n"
    )

    f.write(
        "---------------------------\n"
    )

    f.write(
        "Random Forest: conventional ensemble benchmark "
        "using environmental features.\n"
    )

    f.write(
        "MLP: tabular environmental feature model.\n"
    )

    f.write(
        "CNN: local spatial terrain model using elevation "
        "and slope patches.\n"
    )

    f.write(
        "Transformer: temporal model using the 30-day "
        "rainfall sequence.\n"
    )

    f.write(
        "Fusion: multimodal model combining tabular, "
        "spatial and rainfall representations.\n"
    )


# ============================================================
# CREATE COMPARISON CHART
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

x = range(len(df))

plt.figure(
    figsize=(12, 7)
)

width = 0.15

for i, metric in enumerate(metrics):

    values = df[metric].values

    positions = [
        p + (i - 2) * width
        for p in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )


plt.xticks(
    list(x),
    df["Model"]
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.title(
    "Slidronix Phase G - Model Comparison"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


chart_path = os.path.join(
    OUTPUT_DIR,
    "phase_g_model_comparison.png"
)

plt.savefig(
    chart_path,
    dpi=300
)

plt.close()


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n=================================================="
)

print(
    "SLIDRONIX PHASE G - MODEL COMPARISON"
)

print(
    "==================================================\n"
)

print(
    df.to_string(index=False)
)

print(
    "\nFiles created:"
)

print(
    f"CSV   : {csv_path}"
)

print(
    f"Report: {txt_path}"
)

print(
    f"Chart : {chart_path}"
)

print(
    "\nPHASE G COMPARISON COMPLETE"
)
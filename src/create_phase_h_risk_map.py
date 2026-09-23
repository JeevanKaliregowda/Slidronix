import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# SLIDRONIX PHASE H - GIS RISK MAP
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREDICTIONS = os.path.join(
    BASE_DIR,
    "outputs",
    "phase_h",
    "slidronix_phase_h_risk_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "phase_h"
)

OUTPUT_MAP = os.path.join(
    OUTPUT_DIR,
    "slidronix_phase_h_risk_map.png"
)

OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "phase_h_risk_map_summary.txt"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("SLIDRONIX PHASE H - GIS RISK MAP")
print("=" * 60)

# ------------------------------------------------------------
# 1. LOAD RISK PREDICTIONS
# ------------------------------------------------------------

print("\nLoading Phase H risk predictions...")

df = pd.read_csv(PREDICTIONS)

print(f"Rows loaded: {len(df)}")
print("\nColumns:")
print(df.columns.tolist())

# Required columns
required = [
    "Latitude",
    "Longitude",
    "Landslide_Probability",
    "Risk_Level"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# Remove invalid coordinates
df = df.dropna(
    subset=[
        "Latitude",
        "Longitude",
        "Landslide_Probability"
    ]
).copy()

print(f"Valid mapping points: {len(df)}")

# ------------------------------------------------------------
# 2. RISK COUNTS
# ------------------------------------------------------------

risk_counts = (
    df["Risk_Level"]
    .value_counts()
    .reindex(
        ["Low", "Moderate", "High"],
        fill_value=0
    )
)

print("\nRisk distribution:")
print(risk_counts)

# ------------------------------------------------------------
# 3. CREATE MAP
# ------------------------------------------------------------

print("\nCreating GIS risk map...")

fig, ax = plt.subplots(
    figsize=(13, 11)
)

# ------------------------------------------------------------
# 4. PLOT EACH RISK LEVEL
# ------------------------------------------------------------

# Low
low = df[df["Risk_Level"] == "Low"]

ax.scatter(
    low["Longitude"],
    low["Latitude"],
    s=10,
    alpha=0.55,
    label=f"Low ({len(low)})"
)

# Moderate
moderate = df[df["Risk_Level"] == "Moderate"]

ax.scatter(
    moderate["Longitude"],
    moderate["Latitude"],
    s=14,
    alpha=0.65,
    label=f"Moderate ({len(moderate)})"
)

# High
high = df[df["Risk_Level"] == "High"]

ax.scatter(
    high["Longitude"],
    high["Latitude"],
    s=18,
    alpha=0.75,
    label=f"High ({len(high)})"
)

# ------------------------------------------------------------
# 5. MAP LABELS
# ------------------------------------------------------------

ax.set_title(
    "SLIDRONIX — AI-Based Landslide Risk Map\n"
    "Phase H | Fusion Model Predictions",
    fontsize=18,
    fontweight="bold",
    pad=18
)

ax.set_xlabel(
    "Longitude (°E)",
    fontsize=12
)

ax.set_ylabel(
    "Latitude (°N)",
    fontsize=12
)

# ------------------------------------------------------------
# 6. GRID
# ------------------------------------------------------------

ax.grid(
    True,
    linestyle="--",
    alpha=0.35
)

# ------------------------------------------------------------
# 7. LEGEND
# ------------------------------------------------------------

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=7,
        label=f"Low Risk ({len(low)})"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=8,
        label=f"Moderate Risk ({len(moderate)})"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=9,
        label=f"High Risk ({len(high)})"
    )
]

ax.legend(
    handles=legend_elements,
    title="Predicted Risk Level",
    loc="best",
    frameon=True
)

# ------------------------------------------------------------
# 8. MAP EXTENT
# ------------------------------------------------------------

# Add small padding around study points
lon_min = df["Longitude"].min()
lon_max = df["Longitude"].max()
lat_min = df["Latitude"].min()
lat_max = df["Latitude"].max()

lon_padding = max(
    (lon_max - lon_min) * 0.03,
    0.05
)

lat_padding = max(
    (lat_max - lat_min) * 0.03,
    0.05
)

ax.set_xlim(
    lon_min - lon_padding,
    lon_max + lon_padding
)

ax.set_ylim(
    lat_min - lat_padding,
    lat_max + lat_padding
)

# ------------------------------------------------------------
# 9. INFORMATION BOX
# ------------------------------------------------------------

info_text = (
    f"Prediction samples: {len(df):,}\n"
    f"Low: {len(low):,}\n"
    f"Moderate: {len(moderate):,}\n"
    f"High: {len(high):,}\n"
    f"Mean probability: "
    f"{df['Landslide_Probability'].mean():.4f}\n"
    f"Probability range: "
    f"{df['Landslide_Probability'].min():.4f} – "
    f"{df['Landslide_Probability'].max():.4f}"
)

ax.text(
    0.02,
    0.02,
    info_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round,pad=0.5",
        alpha=0.85
    )
)

# ------------------------------------------------------------
# 10. FOOTNOTE
# ------------------------------------------------------------

fig.text(
    0.5,
    0.015,
    "Risk levels are project-defined visualization categories "
    "based on Fusion-model probability. "
    "Current rainfall input uses the January 1–30, 2022 "
    "reference period.",
    ha="center",
    fontsize=8
)

plt.tight_layout(
    rect=[0, 0.035, 1, 1]
)

# ------------------------------------------------------------
# 11. SAVE MAP
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nRisk map saved:")
print(OUTPUT_MAP)

# ------------------------------------------------------------
# 12. SAVE SUMMARY
# ------------------------------------------------------------

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write("SLIDRONIX PHASE H - GIS RISK MAP SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Prediction samples: {len(df)}\n"
    )

    f.write(
        f"Minimum probability: "
        f"{df['Landslide_Probability'].min():.6f}\n"
    )

    f.write(
        f"Maximum probability: "
        f"{df['Landslide_Probability'].max():.6f}\n"
    )

    f.write(
        f"Mean probability: "
        f"{df['Landslide_Probability'].mean():.6f}\n\n"
    )

    f.write("Risk Distribution\n")
    f.write("-" * 30 + "\n")

    for level in ["Low", "Moderate", "High"]:

        count = int(
            (df["Risk_Level"] == level).sum()
        )

        percentage = (
            count / len(df) * 100
        )

        f.write(
            f"{level}: "
            f"{count} "
            f"({percentage:.2f}%)\n"
        )

    f.write("\n")
    f.write("Risk Classification Thresholds\n")
    f.write("-" * 30 + "\n")
    f.write("Low      : probability < 0.33\n")
    f.write("Moderate : 0.33 <= probability < 0.66\n")
    f.write("High     : probability >= 0.66\n\n")

    f.write("Important limitation\n")
    f.write("-" * 30 + "\n")
    f.write(
        "The current rainfall sequence is a fixed "
        "January 1-30, 2022 reference period. "
        "The map therefore represents model predictions "
        "under the current reference-period inputs, "
        "not a validated real-time operational warning map.\n"
    )

print("\nSummary saved:")
print(OUTPUT_SUMMARY)

print("\n" + "=" * 60)
print("PHASE H STEP 2 COMPLETE")
print("=" * 60)
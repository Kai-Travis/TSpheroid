import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================
# FILES
# ============================================================

FILES = {
    "0 h": r"C:\Github\TSpheroid\Models\predictions_0h.csv",
    "7 h": r"C:\Github\TSpheroid\Models\predictions_7h.csv",
    "0 h + 7 h": r"C:\Github\TSpheroid\Models\predictions_0h_7h.csv",
}


# ============================================================
# LOAD DATA
# ============================================================

results = {}

for model_name, filename in FILES.items():
    df = pd.read_csv(filename)

    print(f"\n{model_name}")
    print(df.columns.tolist())
    print(df.head())

    # Change these names if your CSV uses different column names
    actual = df["actual_LDH"].values
    predicted = df["predicted_LDH"].values

    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    results[model_name] = {
        "actual": actual,
        "predicted": predicted,
        "mae": mae,
        "r2": r2
    }


# ============================================================
# SAME AXIS LIMITS FOR ALL THREE PLOTS
# ============================================================

all_values = []

for result in results.values():
    all_values.extend(result["actual"])
    all_values.extend(result["predicted"])

axis_min = min(all_values)
axis_max = max(all_values)

# Add a little padding
padding = (axis_max - axis_min) * 0.08

axis_min -= padding
axis_max += padding


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 5)
)

for ax, (model_name, result) in zip(axes, results.items()):

    actual = result["actual"]
    predicted = result["predicted"]
    mae = result["mae"]
    r2 = result["r2"]

    # Scatter plot
    ax.scatter(
        actual,
        predicted,
        s=55,
        alpha=0.8
    )

    # Perfect prediction line
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        linewidth=1.5,
        label="Perfect prediction"
    )

    # Same axes for all plots
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    # Labels
    ax.set_xlabel("Actual LDH cytotoxicity")
    ax.set_ylabel("Predicted LDH cytotoxicity")

    ax.set_title(model_name)

    # Metrics inside plot
    ax.text(
        0.05,
        0.95,
        f"MAE = {mae:.3f}\nR² = {r2:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8
        )
    )

    ax.grid(
        True,
        alpha=0.25
    )


# ============================================================
# OVERALL TITLE
# ============================================================

fig.suptitle(
    "Actual vs Predicted LDH Cytotoxicity",
    fontsize=16
)

plt.tight_layout()

# Save high-resolution image for PowerPoint
plt.savefig(
    "actual_vs_predicted_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
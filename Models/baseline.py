from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(r"D:\TrainingData")
DATASET_FILE = ROOT / "metadata" / "dataset_split.csv"

df = pd.read_csv(DATASET_FILE)
print(f"Loaded {len(df)} samples")

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

mean_ldh = train_df["LDH_cytotoxicity"].mean()
print(f"\nMean training LDH: {mean_ldh:.2f}%")

predictions = [mean_ldh] * len(val_df)
actual = val_df["LDH_cytotoxicity"]

mae = mean_absolute_error(actual, predictions)

rmse = mean_squared_error(actual, predictions) ** 0.5

r2 = r2_score(actual, predictions)

print("\n========================================")
print("BASELINE RESULTS")
print("========================================")

print(f"MAE:  {mae:.2f} percentage points")
print(f"RMSE: {rmse:.2f} percentage points")
print(f"R²:   {r2:.3f}")

print(df["LDH_cytotoxicity"].describe())
print(df.groupby("split")["LDH_cytotoxicity"].describe())

import matplotlib.pyplot as plt


# ============================================================
# Plot 1: Overall LDH distribution
# ============================================================

plt.figure()

plt.hist(
    df["LDH_cytotoxicity"],
    bins=15
)

plt.xlabel("LDH cytotoxicity")
plt.ylabel("Number of spheroids")
plt.title("LDH Cytotoxicity Distribution")

plt.show()


# ============================================================
# Plot 2: Train vs validation
# ============================================================

plt.figure()

plt.hist(
    train_df["LDH_cytotoxicity"],
    bins=12,
    alpha=0.6,
    label="Train"
)

plt.hist(
    val_df["LDH_cytotoxicity"],
    bins=12,
    alpha=0.6,
    label="Validation"
)

plt.xlabel("LDH cytotoxicity")
plt.ylabel("Number of spheroids")
plt.title("Train vs Validation LDH Distribution")

plt.legend()

plt.show()
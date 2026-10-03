from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(r"D:\TrainingData")
METADATA_FILE = ROOT / "metadata" / "sample.xlsx"
PROCESSED_ROOT = ROOT / "processed" / "exp01"
OUTPUT_FILE = ROOT / "metadata" / "dataset_split.csv"

RANDOM_SEED = 42
VALIDATION_SIZE = 0.25

df = pd.read_excel(METADATA_FILE)

required_columns = [
    "sample_id",
    "reagent",
    "concentration_µM",
    "Microscope",
    "plate_No.",
    "well",
    "LDH_cytotoxicity",
    "exp_date",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"missing columns in excel: {missing_columns}")

df["sample_id"] = df["sample_id"].astype(str).str.strip()

df["LDH_cytotoxicity"] = pd.to_numeric(df["LDH_cytotoxicity"], errors="coerce")

image_0h = []
image_7h = []

missing_images = []

for sample_id in df["sample_id"]:
    path_0h = (PROCESSED_ROOT / "0h" / f"{sample_id}_cropped.tif")
    path_7h = (PROCESSED_ROOT / "7h" / f"{sample_id}_cropped.tif")

    image_0h.append(str(path_0h))
    image_7h.append(str(path_7h))

    if not path_0h.exists():
        missing_images.append(f"{sample_id}: missing 0h image")

    if not path_7h.exists():
        missing_images.append(f"{sample_id}: missing 7h image")

df["image_0h"] = image_0h
df["image_7h"] = image_7h

missing_ldh = df["LDH_cytotoxicity"].isna()

if missing_ldh.any():
    print("\nWARNING: Missing LDH values:")
    print(df.loc[missing_ldh, "sample_id"].tolist())

if missing_images:
    print("\n warning: missing images")
    for item in missing_images:
        print(" ", item)
    raise FileNotFoundError(f"{len(missing_images)} image files are missing")

else:
    print("all 0h and 7h images were found")

df = df.dropna(subset=["LDH_cytotoxicity"]).reset_index(drop=True)

print(f"\n Samples available for ML: {len(df)}")

train_df, val_df = train_test_split(df, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)

train_df = train_df.copy()
val_df = val_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"

dataset = pd.concat([train_df, val_df], ignore_index=True)

dataset["experiment_id"] = "exp01"

columns = [
    "sample_id",
    "experiment_id",
    "image_0h",
    "image_7h",
    "LDH_cytotoxicity",
    "reagent",
    "concentration_µM",
    "Microscope",
    "plate_No.",
    "well",
    "exp_date",
    "split",
]

dataset = dataset[columns]

dataset.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Dataset split complete")

print(f"Total:      {len(dataset)}")
print(f"Train:      {(dataset['split'] == 'train').sum()}")
print(f"Validation: {(dataset['split'] == 'val').sum()}")

print("\nLDH statistics:")

print(
    dataset
    .groupby("split")["LDH_cytotoxicity"]
    .agg(["count", "mean", "std", "min", "max"])
)

print("\nSaved to:")
print(OUTPUT_FILE)
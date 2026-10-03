from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(r"D:\TrainingData")
METDADATA_FILE = ROOT / "metadata" / "sample.xlsx"
PROCESSED_ROOT = ROOT / "processed"

df = pd.read_excel(METDADATA_FILE)

print("=" * 60)
print("DATASET CHECK")
print("=" * 60)

print(f"\nExcel rows: {len(df)}")

print("\nColumns:")
print(list(df.columns))

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
    print("\nError: missing columns:")
    for col in missing_columns:
        print(f" - {col}")
    raise SystemExit

duplicates = df[df["sample_id"].duplicated(keep=False)]

if len(duplicates) > 0:
    print("\nWarning: duplicate sample ids;")

    print(duplicates[
        ["sample_id", "reagent",
         "concentration_µM",
         "LDH_cytotoxicity"]
    ].to_string(index=False)
    )
else:
    print("\n no duplicate sample ids")

missing_ldh = df[df["LDH_cytotoxicity"].isna()]

if len(missing_ldh) > 0:
    print(f"\nwarning: {len(missing_ldh)} samples have missing LDH values")
    print(missing_ldh["sample_id"].to_list())

else:
    print("\n no missing ldh values")

print("\n" + "=" * 60)
print("IMAGE CHECK")
print("=" * 60)

missing_0h = []
missing_7h = []

wrong_size = []
wrong_mode = []

checked = 0

for _, row in df.iterrows():
    sample_id = str(row["sample_id"])

    path_0h = (PROCESSED_ROOT / "exp01" / "0h" / f"{sample_id}_cropped.tif")

    path_7h = (PROCESSED_ROOT / "exp01" / "7h" / f"{sample_id}_cropped.tif")

    if not path_0h.exists():
        missing_0h.append(sample_id)

    else:
        try:
            with Image.open(path_0h) as img:
                checked += 1
                if img.size != (650, 650):
                    wrong_size.append((sample_id, "0h", img.size))
                if img.mode not in ["I;16", "I"]:
                    wrong_mode.append((sample_id, "0h", img.mode))

        except Exception as e:
            print(f"Error reading {path_0h}: {e}")

    if not path_7h.exists():
        missing_7h.append(sample_id)
    else:
        try:
            with Image.open(path_7h) as img:
                checked += 1
                if img.size != (650, 650):
                    wrong_size.append((sample_id, "7h", img.size))
                if img.mode not in ["I;16", "I"]:
                    wrong_mode.append((sample_id, "7h", img.mode))

        except Exception as e:
            print(f"Error reading {path_7h}: {e}")


print(f"\nImages checked: {checked}")


if missing_0h:
    print(f"\n Missing 0h images: {len(missing_0h)}")
    print(missing_0h)

else:
    print("\nAll 0h images found")


if missing_7h:
    print(f"\nMissing 7h images: {len(missing_7h)}")
    print(missing_7h)

else:
    print("\nAll 7h images found")


if wrong_size:
    print(f"\nImages with wrong dimensions:{len(wrong_size)}")
    for item in wrong_size:
        print(item)

else:
    print("\n All images are 650 × 650")


if wrong_mode:
    print(f"\nImages with unexpected image mode: {len(wrong_mode)}"
    )
    for item in wrong_mode:
        print(item)

else:
    print("\n Image bit depth/mode looks correct")

    print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Samples in Excel: {len(df)}")
print(f"Missing 0h:       {len(missing_0h)}")
print(f"Missing 7h:       {len(missing_7h)}")
print(f"Wrong size:       {len(wrong_size)}")
print(f"Wrong image mode: {len(wrong_mode)}")
print(f"Missing LDH:      {len(missing_ldh)}")

if (
    len(missing_0h) == 0
    and len(missing_7h) == 0
    and len(wrong_size) == 0
    and len(missing_ldh) == 0
):

    print("\nDataset looks ready for the next step!")

else:
    print(
        "\nFix the problems above before training."
    )
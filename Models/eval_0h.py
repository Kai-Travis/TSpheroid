from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# Settings
# ============================================================

ROOT = Path(r"D:\TrainingData")
CSV_FILE = ROOT / "metadata" / "dataset_split.csv"

MODEL_FILE = Path("best_resnet18_0h.pth")

BATCH_SIZE = 8

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# ============================================================
# Dataset
# ============================================================

class SpheroidDataset(Dataset):

    def __init__(self, csv_file, split):

        self.df = pd.read_csv(csv_file)

        self.df = self.df[
            self.df["split"] == split
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = Path(row["image_0h"])

        image = Image.open(image_path).convert("L")

        image = transform(image)

        # Grayscale → 3 channels
        image = image.repeat(3, 1, 1)

        # ImageNet normalization
        image = normalize(image)

        label = torch.tensor(
            row["LDH_cytotoxicity"],
            dtype=torch.float32
        )

        sample_id = row["sample_id"]

        return image, label, sample_id


# ============================================================
# Validation dataset
# ============================================================

dataset = SpheroidDataset(
    CSV_FILE,
    "val"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Validation samples:", len(dataset))


# ============================================================
# Load model
# ============================================================

model = models.resnet18(weights="DEFAULT")

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ============================================================
# Make predictions
# ============================================================

results = []

with torch.no_grad():

    for images, labels, sample_ids in loader:

        images = images.to(device)

        outputs = model(images).squeeze(1)

        predictions = outputs.cpu().numpy()
        labels = labels.numpy()

        for sample_id, actual, predicted in zip(
            sample_ids,
            labels,
            predictions
        ):

            error = abs(predicted - actual)

            results.append({
                "sample_id": sample_id,
                "actual_LDH": actual,
                "predicted_LDH": predicted,
                "absolute_error": error
            })


# ============================================================
# Results
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "actual_LDH"
).reset_index(drop=True)


mae = results_df["absolute_error"].mean()

rmse = np.sqrt(
    np.mean(
        (
            results_df["actual_LDH"]
            - results_df["predicted_LDH"]
        ) ** 2
    )
)

ss_res = np.sum(
    (
        results_df["actual_LDH"]
        - results_df["predicted_LDH"]
    ) ** 2
)

ss_tot = np.sum(
    (
        results_df["actual_LDH"]
        - results_df["actual_LDH"].mean()
    ) ** 2
)

r2 = 1 - (ss_res / ss_tot)


# ============================================================
# Print results
# ============================================================

print()
print("RESULTS")
print("=" * 60)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

print()
print("Prediction range:")
print(
    f"Actual:    "
    f"{results_df['actual_LDH'].min():.3f}"
    f" → "
    f"{results_df['actual_LDH'].max():.3f}"
)

print(
    f"Predicted: "
    f"{results_df['predicted_LDH'].min():.3f}"
    f" → "
    f"{results_df['predicted_LDH'].max():.3f}"
)

print()
print("Predictions:")
print(results_df.to_string(index=False))


# ============================================================
# Save CSV
# ============================================================

OUTPUT_FILE = Path("predictions_0h.csv")

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:", OUTPUT_FILE)
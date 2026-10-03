from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

ROOT = Path(r"D:\TrainingData")
CSV_FILE = ROOT / "metadata" / "dataset_split.csv"

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001

MODEL_FILE = Path("best_resnet18_0h_7h.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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
        image_0h_path = Path(row["image_0h"])
        image_0h = Image.open(image_0h_path).convert("L")
        image_0h = transform(image_0h)
        # Grayscale → 3 channels
        image_0h = image_0h.repeat(3, 1, 1)

        # ImageNet normalization
        image_0h = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image_0h)

        image_7h_path = Path(row["image_7h"])
        image_7h = Image.open(image_7h_path).convert("L")
        image_7h = transform(image_7h)
        # Grayscale → 3 channels
        image_7h = image_7h.repeat(3, 1, 1)

        # ImageNet normalization
        image_7h = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image_7h)

        label = torch.tensor(
            row["LDH_cytotoxicity"],
            dtype=torch.float32
        )

        return image_0h, image_7h, label

train_dataset = SpheroidDataset(
    CSV_FILE,
    "train"
)

val_dataset = SpheroidDataset(
    CSV_FILE,
    "val"
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

resnet = models.resnet18(weights="DEFAULT")

resnet.fc = nn.Identity()

for parameter in resnet.parameters():
    parameter.requires_grad = False

resnet = resnet.to(device)

regression_head = nn.Linear(1024, 1)

regression_head = regression_head.to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    regression_head.parameters(), lr = LEARNING_RATE
)

def evaluate(resnet, regression_head, loader):
    resnet.eval()
    regression_head.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for images_0h, images_7h, labels in loader:
            images_0h = images_0h.to(device)
            images_7h = images_7h.to(device)
            labels = labels.to(device)

            features_0h = resnet(images_0h)
            features_7h = resnet(images_7h)

            combined_features = torch.cat([features_0h, features_7h], dim=1)

            outputs = regression_head(combined_features).squeeze(1)

            predictions.extend(outputs.cpu().numpy())

            targets.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    mae = np.mean(np.abs(predictions - targets))

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    ss_res = np.sum((targets - predictions) ** 2)

    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    return mae, rmse, r2


best_mae = float("inf")

print()
print("Starting training...")
print()

for epoch in range(EPOCHS):
    resnet.eval()
    regression_head.train()
    running_loss = 0.0

    for images_0h, images_7h, labels in train_loader:
        images_0h = images_0h.to(device)
        images_7h = images_7h.to(device)
        labels = labels.to(device)

        features_0h = resnet(images_0h)
        features_7h = resnet(images_7h)

        combined_features = torch.cat([features_0h, features_7h], dim=1)

        outputs = regression_head(combined_features).squeeze(1)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += (loss.item() * images_0h.size(0))

    train_loss = (running_loss / len(train_dataset))

    val_mae, val_rmse, val_r2 = evaluate(
        resnet, regression_head, val_loader
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val MAE: {val_mae:.4f} | "
        f"Val RMSE: {val_rmse:.4f} | "
        f"Val R²: {val_r2:.4f}"
    )

    if val_mae < best_mae:
        best_mae = val_mae
        torch.save(regression_head.state_dict(), MODEL_FILE)

        print(
            f"  → New best model saved "
            f"(MAE = {best_mae:.4f})"
        )

print()
print("Training finished.")
print("Best validation MAE:", best_mae)
print("Saved model:", MODEL_FILE)
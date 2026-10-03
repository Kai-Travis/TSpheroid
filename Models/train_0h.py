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

MODEL_FILE = Path("best_resnet18_0h.pth")

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
        image_path = Path(row["image_0h"])
        image = Image.open(image_path).convert("L")
        image = transform(image)
        # Grayscale → 3 channels
        image = image.repeat(3, 1, 1)

        # ImageNet normalization
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)
        label = torch.tensor(
            row["LDH_cytotoxicity"],
            dtype=torch.float32
        )
        return image, label


train_dataset = SpheroidDataset(
    CSV_FILE,
    "train"
)

val_dataset = SpheroidDataset(
    CSV_FILE,
    "val"
)

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

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


model = models.resnet18(weights="DEFAULT")


# Freeze pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False


# Replace final layer
model.fc = nn.Linear(
    model.fc.in_features,
    1
)

model = model.to(device)


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

def evaluate(model, loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)

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

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images).squeeze(1)
        # Calculate error
        loss = criterion(outputs, labels)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    train_loss = (
        running_loss / len(train_dataset)
    )

    # Validation
    val_mae, val_rmse, val_r2 = evaluate(
        model,
        val_loader
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val MAE: {val_mae:.4f} | "
        f"Val RMSE: {val_rmse:.4f} | "
        f"Val R²: {val_r2:.4f}"
    )

    # Save best model
    if val_mae < best_mae:
        best_mae = val_mae
        torch.save(
            model.state_dict(),
            MODEL_FILE
        )
        print(
            f"  → New best model saved "
            f"(MAE = {best_mae:.4f})"
        )


print()
print("Training finished.")
print("Best validation MAE:", best_mae)
print("Saved model:", MODEL_FILE)
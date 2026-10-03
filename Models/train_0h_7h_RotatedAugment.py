from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
import torchvision.transforms.functional as TF


ROOT = Path(r"D:\TrainingData")

CSV_FILE = ROOT / "metadata" / "dataset_split.csv"

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001

MODEL_FILE = Path("best_resnet18_0h_7h_rotation.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class SpheroidDataset(Dataset):

    def __init__(self, csv_file, split, augmentation=False):

        self.df = pd.read_csv(csv_file)

        self.df = self.df[
            self.df["split"] == split
        ].reset_index(drop=True)

        self.augmentation = augmentation

        # Original samples
        self.samples = []

        for idx in range(len(self.df)):
            self.samples.append(
                (idx, False)
            )

        # Add ONE rotated copy of every training sample
        if augmentation:

            for idx in range(len(self.df)):
                self.samples.append(
                    (idx, True)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        original_idx, rotate = self.samples[idx]

        row = self.df.iloc[original_idx]

        image_0h_path = Path(
            row["image_0h"]
        )

        image_7h_path = Path(
            row["image_7h"]
        )

        image_0h = Image.open(
            image_0h_path
        ).convert("L")

        image_7h = Image.open(
            image_7h_path
        ).convert("L")

        # --------------------------------------------------
        # Rotate both timepoints by EXACTLY the same angle
        # --------------------------------------------------

        if rotate:

            angle = random.uniform(
                -180,
                180
            )

            image_0h = TF.rotate(
                image_0h,
                angle
            )

            image_7h = TF.rotate(
                image_7h,
                angle
            )

        # --------------------------------------------------
        # Resize
        # --------------------------------------------------

        image_0h = base_transform(image_0h)
        image_7h = base_transform(image_7h)

        # --------------------------------------------------
        # Grayscale → 3 channels
        # --------------------------------------------------

        image_0h = image_0h.repeat(
            3, 1, 1
        )

        image_7h = image_7h.repeat(
            3, 1, 1
        )

        # --------------------------------------------------
        # ImageNet normalization
        # --------------------------------------------------

        image_0h = normalize(image_0h)
        image_7h = normalize(image_7h)

        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        label = torch.tensor(
            row["LDH_cytotoxicity"],
            dtype=torch.float32
        )

        return (
            image_0h,
            image_7h,
            label
        )


# --------------------------------------------------
# Datasets
# --------------------------------------------------

train_dataset = SpheroidDataset(
    CSV_FILE,
    "train",
    augmentation=True
)

val_dataset = SpheroidDataset(
    CSV_FILE,
    "val",
    augmentation=False
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


print(
    "Original training samples:",
    len(train_dataset.df)
)

print(
    "Augmented training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# --------------------------------------------------
# ResNet feature extractors
# --------------------------------------------------

resnet_0h = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

resnet_7h = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# Remove classification layer
resnet_0h.fc = nn.Identity()
resnet_7h.fc = nn.Identity()


# Freeze pretrained layers
for parameter in resnet_0h.parameters():
    parameter.requires_grad = False

for parameter in resnet_7h.parameters():
    parameter.requires_grad = False


# --------------------------------------------------
# Combined model
# --------------------------------------------------

class TwoImageModel(nn.Module):

    def __init__(
        self,
        resnet_0h,
        resnet_7h
    ):

        super().__init__()

        self.resnet_0h = resnet_0h
        self.resnet_7h = resnet_7h

        self.regression_head = nn.Linear(
            1024,
            1
        )

    def forward(
        self,
        image_0h,
        image_7h
    ):

        features_0h = self.resnet_0h(
            image_0h
        )

        features_7h = self.resnet_7h(
            image_7h
        )

        combined = torch.cat(
            [
                features_0h,
                features_7h
            ],
            dim=1
        )

        output = self.regression_head(
            combined
        )

        return output


model = TwoImageModel(
    resnet_0h,
    resnet_7h
)

model = model.to(device)


# --------------------------------------------------
# Loss + optimizer
# --------------------------------------------------

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.regression_head.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, loader):

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for (
            image_0h,
            image_7h,
            labels
        ) in loader:

            image_0h = image_0h.to(device)
            image_7h = image_7h.to(device)
            labels = labels.to(device)

            outputs = model(
                image_0h,
                image_7h
            ).squeeze(1)

            predictions.extend(
                outputs.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )

    predictions = np.array(predictions)
    targets = np.array(targets)

    mae = np.mean(
        np.abs(
            predictions - targets
        )
    )

    rmse = np.sqrt(
        np.mean(
            (predictions - targets) ** 2
        )
    )

    ss_res = np.sum(
        (targets - predictions) ** 2
    )

    ss_tot = np.sum(
        (targets - np.mean(targets)) ** 2
    )

    r2 = 1 - (
        ss_res / ss_tot
    )

    return mae, rmse, r2


# --------------------------------------------------
# Training
# --------------------------------------------------

best_mae = float("inf")

print()
print("Starting training...")
print()

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for (
        image_0h,
        image_7h,
        labels
    ) in train_loader:

        image_0h = image_0h.to(device)
        image_7h = image_7h.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(
            image_0h,
            image_7h
        ).squeeze(1)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * image_0h.size(0)
        )

    train_loss = (
        running_loss
        / len(train_dataset)
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

print(
    "Best validation MAE:",
    best_mae
)

print(
    "Saved model:",
    MODEL_FILE
)
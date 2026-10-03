from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(r"D:\TrainingData")
CSV_FILE = ROOT / "metadata" / "dataset_split.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
class SpheroidDataSet(Dataset):
    def __init__(self, csv_file, split):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row["image_7h"])
        image = Image.open(image_path).convert("L")
        image = self.transform(image)
        image = image.repeat(3, 1, 1)

        label = torch.tensor(row["LDH_cytotoxicity"], dtype = torch.float32)

        return image, label

dataset = SpheroidDataSet(CSV_FILE, split="train")

print("number of training samples:", len(dataset))

image, label = dataset[0]
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("LDH:", label)
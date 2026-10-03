from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# Run this script from:
# C:\Github\TSpheroid\Models

MODEL_FILE = Path(r"C:\Github\TSpheroid\Models\best_resnet18_0h_7h.pth")

DATASET_FILE = Path(
    r"D:\TrainingData\metadata\dataset_split.csv"
)

OUTPUT_FOLDER = Path(r"C:\Github\TSpheroid\GRADCAMResults")
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Put the sample_id you want to inspect here.
# Example:
SAMPLE_ID = "EXP1_050"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ============================================================
# RESNET18 BACKBONE
# ============================================================

def create_backbone():
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Remove classification layer
    backbone.fc = nn.Identity()

    return backbone


# ============================================================
# TWO-BRANCH MODEL
# ============================================================

class TwoBranchResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone_0h = create_backbone()
        self.backbone_7h = create_backbone()

        self.regression_head = nn.Linear(512 * 2, 1)

    def forward(self, image_0h, image_7h):

        features_0h = self.backbone_0h(image_0h)
        features_7h = self.backbone_7h(image_7h)

        combined = torch.cat(
            [features_0h, features_7h],
            dim=1
        )

        output = self.regression_head(combined)

        return output


# ============================================================
# LOAD MODEL
# ============================================================

model = TwoBranchResNet()

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

# Your training script saved only the regression head.
model.regression_head.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# FIND SAMPLE
# ============================================================

df = pd.read_csv(DATASET_FILE)

row = df[df["sample_id"] == SAMPLE_ID]

if len(row) == 0:
    raise ValueError(
        f"Could not find sample_id: {SAMPLE_ID}"
    )

row = row.iloc[0]

image_0h_path = Path(row["image_0h"])
image_7h_path = Path(row["image_7h"])

actual_ldh = row["LDH_cytotoxicity"]

print()
print("Sample:", SAMPLE_ID)
print("0 h:", image_0h_path)
print("7 h:", image_7h_path)
print("Actual LDH:", actual_ldh)


# ============================================================
# LOAD IMAGES
# ============================================================

original_0h = Image.open(image_0h_path).convert("L")
original_7h = Image.open(image_7h_path).convert("L")

image_0h = transform(original_0h)
image_0h = image_0h.repeat(3, 1, 1)
image_0h = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)(image_0h)
image_0h = image_0h.unsqueeze(0).to(device)

image_7h = transform(original_7h)
image_7h = image_7h.repeat(3, 1, 1)
image_7h = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)(image_7h)
image_7h = image_7h.unsqueeze(0).to(device)


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):

        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    def generate(self, output):

        # Clear previous gradients
        self.model.zero_grad()

        # Backpropagate from predicted LDH
        output.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        # Average gradients over spatial dimensions
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination of feature maps
        cam = (weights * activations).sum(dim=1)

        # ReLU
        cam = torch.relu(cam)

        # Remove batch dimension
        cam = cam[0]

        # Normalize 0 → 1
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.detach().cpu().numpy()


# ============================================================
# CREATE GRAD-CAM OBJECTS
# ============================================================

# Last convolutional layer of each ResNet
target_layer_0h = model.backbone_0h.layer4[-1].conv2
target_layer_7h = model.backbone_7h.layer4[-1].conv2

gradcam_0h = GradCAM(
    model,
    target_layer_0h
)

gradcam_7h = GradCAM(
    model,
    target_layer_7h
)


# ============================================================
# FORWARD PASS
# ============================================================

model.zero_grad()

prediction = model(
    image_0h,
    image_7h
)

predicted_ldh = prediction.item()

print("Predicted LDH:", predicted_ldh)


# ============================================================
# GRAD-CAM FOR 0 h
# ============================================================

cam_0h = gradcam_0h.generate(prediction)


# ============================================================
# GRAD-CAM FOR 7 h
# ============================================================

# We need another forward pass because the previous backward
# pass consumed the computation graph.

model.zero_grad()

prediction = model(
    image_0h,
    image_7h
)

cam_7h = gradcam_7h.generate(prediction)


# ============================================================
# PREPARE ORIGINAL IMAGES
# ============================================================

original_0h_np = np.array(
    original_0h.resize((224, 224)),
    dtype=np.float32
) / 255.0

original_7h_np = np.array(
    original_7h.resize((224, 224)),
    dtype=np.float32
) / 255.0


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10)
)


# -------------------------
# 0 h original
# -------------------------

axes[0, 0].imshow(
    original_0h_np,
    cmap="gray"
)

axes[0, 0].set_title("0 h — Original")
axes[0, 0].axis("off")


# -------------------------
# 0 h Grad-CAM
# -------------------------

axes[0, 1].imshow(
    original_0h_np,
    cmap="gray"
)

axes[0, 1].imshow(
    cam_0h,
    cmap="jet",
    alpha=0.45
)

axes[0, 1].set_title("0 h — Grad-CAM")
axes[0, 1].axis("off")


# -------------------------
# 7 h original
# -------------------------

axes[1, 0].imshow(
    original_7h_np,
    cmap="gray"
)

axes[1, 0].set_title("7 h — Original")
axes[1, 0].axis("off")


# -------------------------
# 7 h Grad-CAM
# -------------------------

axes[1, 1].imshow(
    original_7h_np,
    cmap="gray"
)

axes[1, 1].imshow(
    cam_7h,
    cmap="jet",
    alpha=0.45
)

axes[1, 1].set_title("7 h — Grad-CAM")
axes[1, 1].axis("off")


fig.suptitle(
    f"{SAMPLE_ID}\n"
    f"Actual LDH = {actual_ldh:.3f}    "
    f"Predicted LDH = {predicted_ldh:.3f}",
    fontsize=14
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_file = OUTPUT_FOLDER / f"{SAMPLE_ID}_gradcam.png"

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print()
print("Saved:", output_file)
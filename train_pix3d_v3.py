
import os
import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from models.rgb2point import RGB2Point
from data_loaders.pix3d_dataset import Pix3DDataset
from utils.geometry_loss import geometry_loss

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Configuration
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Training device:", device, flush=True)

# Match preprocessing used by the existing model.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def create_loader(split, shuffle):
    dataset = Pix3DDataset(
        images_root="datasets/pix3d/images/",
        annotations_file=f"pix3d_{split}.json",
        points_root="datasets/pix3d/points/",
        transform=transform,
        num_points=1024,
        category="wardrobe"
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )

train_loader = create_loader("train", True)
val_loader = create_loader("val", False)

model = RGB2Point().to(device)

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE
)

os.makedirs("outputs", exist_ok=True)
checkpoint_path = "outputs/pix3d_v3_best_model.pth"
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_total = 0.0
    train_count = 0

    for images, gt in train_loader:
        images = images.to(device)
        gt = gt.to(device).float()

        optimizer.zero_grad()
        pred = model(images)
        loss, cd, bbox = geometry_loss(pred, gt, bbox_weight=0.1)

        loss.backward()
        optimizer.step()

        train_total += loss.item() * images.size(0)
        train_count += images.size(0)

    train_loss = train_total / train_count

    model.eval()
    val_total = 0.0
    val_count = 0

    with torch.no_grad():
        for images, gt in val_loader:
            images = images.to(device)
            gt = gt.to(device).float()

            pred = model(images)
            loss, cd, bbox = geometry_loss(pred, gt, bbox_weight=0.1)

            val_total += loss.item() * images.size(0)
            val_count += images.size(0)

    val_loss = val_total / val_count

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Geometry Loss: {train_loss:.6f} | "
        f"Val Geometry Loss: {val_loss:.6f}",
        flush=True
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)
        print("Saved new best checkpoint!", flush=True)

print("Training complete!")
print("Best validation geometry loss:", best_val_loss)
print("Checkpoint:", checkpoint_path)


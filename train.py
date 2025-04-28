import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.rgb2point import RGB2Point
from datasets.shapenet_dataset import ShapeNetDataset
from utils.chamfer_distance import chamfer_distance
import os

# Config
batch_size = 8
epochs = 10
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ShapeNetDataset(
    images_root="datasets/shapenet/images/",
    points_root="datasets/shapenet/points/",
    transform=transform,
    num_points=1024
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = RGB2Point().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Output folder
os.makedirs('outputs', exist_ok=True)

# Training Loop
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, gt_pointclouds in train_loader:
        images = images.to(device)
        gt_pointclouds = gt_pointclouds.to(device)

        optimizer.zero_grad()
        pred_pointclouds = model(images)
        loss = chamfer_distance(pred_pointclouds, gt_pointclouds)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Chamfer Loss: {avg_loss:.6f}")

    # Save the best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'outputs/best_model.pth')
        print(f"✅ Best model saved at Epoch {epoch+1}!")

print("Training Done ✅")

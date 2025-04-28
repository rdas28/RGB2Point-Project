import numpy as np
import torch
from PIL import Image
import os

# Create folders
os.makedirs('datasets/shapenet/images', exist_ok=True)
os.makedirs('datasets/shapenet/points', exist_ok=True)

# Generate 5 dummy images and pointclouds
for i in range(5):
    # Create a random RGB image (224x224)
    img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(f'datasets/shapenet/images/dummy_{i}.png')

    # Create a random point cloud (1024 x 3 points)
    points = np.random.rand(1024, 3).astype(np.float32)
    np.save(f'datasets/shapenet/points/dummy_{i}.npy', points)

print("✅ Dummy ShapeNet dataset created: 5 images + 5 point clouds!")

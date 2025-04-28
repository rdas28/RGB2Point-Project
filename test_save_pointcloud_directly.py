import torch
import numpy as np
from utils.visualize_pointcloud import save_pointcloud
import os

os.makedirs('outputs/predictions/', exist_ok=True)

# Create a dummy point cloud (1024 points, 3D)
dummy_points = np.random.rand(1024, 3).astype(np.float32)

# Save the dummy point cloud
save_pointcloud(torch.tensor(dummy_points), 'outputs/predictions/test_dummy.ply')

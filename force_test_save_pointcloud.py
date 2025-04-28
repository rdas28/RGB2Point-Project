import torch
import numpy as np
import os
from utils.visualize_pointcloud import save_pointcloud

os.makedirs('outputs/predictions/', exist_ok=True)

# Create dummy points
dummy_points = torch.rand(1024, 3)

# Save
save_pointcloud(dummy_points, 'outputs/predictions/force_test_pred.ply')

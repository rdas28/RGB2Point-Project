import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def save_pointcloud(points, filename):
    """
    Save predicted points as a .ply file.
    Args:
        points: Tensor (Nx3) or numpy array
        filename: path to save .ply
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if points.ndim != 2 or points.shape[1] != 3:
        print(f"❌ Cannot save {filename}: wrong shape {points.shape}")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    success = o3d.io.write_point_cloud(filename, pcd)

    if success:
        print(f"✅ Saved point cloud: {filename}")
    else:
        print(f"❌ Failed to save point cloud: {filename}")

def visualize_input_output(input_image, predicted_points, save_path):
    """
    Visualize the input RGB image and predicted 3D point cloud side-by-side.
    """
    fig = plt.figure(figsize=(10, 5))

    # Left side: RGB input
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(input_image.permute(1, 2, 0).cpu().numpy())
    ax1.set_title('Input RGB Image')
    ax1.axis('off')

    # Right side: Predicted point cloud
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if isinstance(predicted_points, torch.Tensor):
        pred_np = predicted_points.detach().cpu().numpy()
    else:
        pred_np = predicted_points

    ax2.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], s=1)
    ax2.set_title('Predicted 3D Point Cloud')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved visualization: {save_path}")

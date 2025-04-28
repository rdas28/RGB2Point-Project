import open3d as o3d
import imageio
import numpy as np
import os

def generate_rotating_gif(pointcloud_path, gif_save_path, n_frames=36):
    """
    Args:
        pointcloud_path (str): Path to .ply file.
        gif_save_path (str): Where to save the output GIF.
        n_frames (int): Number of frames for full 360 rotation.
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    
    # Set up visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)

    images = []
    for i in range(n_frames):
        R = pcd.get_rotation_matrix_from_axis_angle([0, 2 * np.pi * i / n_frames, 0])
        pcd.rotate(R, center=(0, 0, 0))
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        img = vis.capture_screen_float_buffer(False)
        img = (255 * np.asarray(img)).astype(np.uint8)
        images.append(img)

    vis.destroy_window()

    # Save GIF
    imageio.mimsave(gif_save_path, images, duration=0.05)
    print(f"✅ Saved rotating GIF: {gif_save_path}")

# ---------- Example usage below ----------

# Generate GIFs for first 5 predictions
os.makedirs('outputs/predictions/', exist_ok=True)

for idx in range(5):
    pointcloud_path = f'outputs/predictions/pred_{idx}.ply'
    gif_save_path = f'outputs/predictions/rotate_pred_{idx}.gif'
    if os.path.exists(pointcloud_path):
        generate_rotating_gif(pointcloud_path, gif_save_path)

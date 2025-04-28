import os
import numpy as np
import trimesh

models_root = 'model/wardrobe/'  # corrected path
points_root = 'datasets/pix3d/points/'
os.makedirs(points_root, exist_ok=True)

model_names = [name for name in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, name))]

for model_name in model_names:
    obj_path = os.path.join(models_root, model_name, 'model.obj')
    save_path = os.path.join(points_root, model_name + '.npy')

    if os.path.exists(obj_path):
        mesh = trimesh.load(obj_path)

        # 🔥 Fix for Scene loading
        if isinstance(mesh, trimesh.Scene):
            print(f"[INFO] {model_name} loaded as Scene — merging geometries")
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

        # Sample points
        sampled_points, _ = trimesh.sample.sample_surface(mesh, 1024)
        np.save(save_path, sampled_points.astype(np.float32))
        print(f"✅ Real point cloud saved at {save_path}")
    else:
        print(f"❌ Model file not found: {obj_path}")

print("✅ All real .npy point clouds generated!")

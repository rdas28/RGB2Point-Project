import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class Pix3DDataset(Dataset):
    def __init__(self, images_root, annotations_file, points_root, transform=None, num_points=1024, category='chair'):
        """
        Args:
            images_root (str): Path to folder containing RGB images.
            annotations_file (str): Path to pix3d.json metadata.
            points_root (str): Path to folder containing .npy point clouds.
            transform (callable, optional): Transformations for input images.
            num_points (int): Number of points to sample from each point cloud.
            category (str): Which category to filter (e.g., 'chair', 'wardrobe').
        """
        self.images_root = images_root
        self.points_root = points_root
        self.transform = transform
        self.num_points = num_points
        self.category = category

        # Load and filter annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.samples = []
        for ann in self.annotations:
            if 'model' in ann and ann['model'] is not None:
                if 'category' in ann and ann['category'] == self.category:
                    self.samples.append(ann)

        print(f"✅ Pix3DDataset initialized with {len(self.samples)} samples for category '{self.category}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 🔥 Image loading
        img_rel_path = sample['img']
        if img_rel_path.startswith('img/'):
            img_rel_path = img_rel_path[4:]
        img_path = os.path.join(self.images_root, img_rel_path)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # 🔥 Point cloud loading
        # Correct extraction of model name
        model_name = sample['model'].split('/')[-2] + '.npy'  # <-- 🔥 FIXED HERE
        pc_path = os.path.join(self.points_root, model_name)

        pointcloud = np.load(pc_path)
        pointcloud = torch.from_numpy(pointcloud).float()

        # 🔥 Sample or repeat points to maintain num_points
        if pointcloud.size(0) > self.num_points:
            indices = torch.randperm(pointcloud.size(0))[:self.num_points]
            pointcloud = pointcloud[indices]
        elif pointcloud.size(0) < self.num_points:
            repeat_factor = (self.num_points + pointcloud.size(0) - 1) // pointcloud.size(0)
            pointcloud = pointcloud.repeat(repeat_factor, 1)[:self.num_points]

        return image, pointcloud

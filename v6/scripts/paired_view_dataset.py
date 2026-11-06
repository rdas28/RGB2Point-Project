import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class PairedViewDataset(Dataset):

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent.parent

        with self.manifest_path.open() as f:
            records = json.load(f)

        grouped = defaultdict(list)

        for record in records:
            grouped[record["object_id"]].append(record)

        self.objects = []

        for object_id, views in sorted(grouped.items()):
            views = sorted(views, key=lambda r: int(r["view_id"]))

            if len(views) < 2:
                raise ValueError(
                    f"Object {object_id} has fewer than 2 views"
                )

            point_paths = {r["points"] for r in views}

            if len(point_paths) != 1:
                raise ValueError(
                    f"Inconsistent ground truth for {object_id}"
                )

            self.objects.append((object_id, views))

        self.transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.objects) * 4

    def load_image(self, record):
        image_path = self.root / record["image"]

        with Image.open(image_path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, idx):
        object_id, views = self.objects[idx // 4]

        pair_index = idx % 4
        view_indices = list(range(len(views)))

        if pair_index == 0:
            selected = view_indices[0:2]
        elif pair_index == 1:
            selected = view_indices[2:4]
        elif pair_index == 2:
            selected = view_indices[4:6]
        else:
            selected = view_indices[6:8]

        record_a, record_b = [views[i] for i in selected]

        if random.random() < 0.5:
            record_a, record_b = record_b, record_a

        image_a = self.load_image(record_a)
        image_b = self.load_image(record_b)

        points_path = self.root / record_a["points"]
        points = np.load(points_path).astype(np.float32)

        if points.shape != (1024, 3):
            raise ValueError(
                f"Invalid point cloud shape: {points.shape}"
            )

        if not np.isfinite(points).all():
            raise ValueError(
                f"Non-finite point cloud: {object_id}"
            )

        return {
            "image_a": image_a,
            "image_b": image_b,
            "points": torch.from_numpy(points),
            "object_id": object_id,
            "view_a": int(record_a["view_id"]),
            "view_b": int(record_b["view_id"])
        }

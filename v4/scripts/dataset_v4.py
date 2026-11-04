import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RGB2PointV4Dataset(Dataset):
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent.parent

        with self.manifest_path.open() as f:
            self.records = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        image_path = self.root / record["image"]
        points_path = self.root / record["points"]

        with Image.open(image_path) as image:
            image = self.transform(image.convert("RGB"))

        points = np.load(points_path)

        if points.shape != (1024, 3):
            raise ValueError(
                f"Expected (1024, 3), got {points.shape}"
            )

        points = torch.from_numpy(
            points.astype(np.float32)
        )

        return {
            "image": image,
            "points": points,
            "object_id": record["object_id"],
            "category": record["category"],
            "view_id": record["view_id"],
        }

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from v4.models.rgb2point_v4 import RGB2PointV4
from v4.models.geometry_aware_loss import geometry_aware_loss
from v4.scripts.dataset_v4 import RGB2PointV4Dataset


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dimension-weight",
    type=float,
    required=True
)

parser.add_argument(
    "--name",
    type=str,
    required=True
)

args = parser.parse_args()

SEED = 42
EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
COVERAGE_WEIGHT = 2.0
DIMENSION_WEIGHT = args.dimension_weight

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

output_dir = Path("v4/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint = output_dir / f"{args.name}_best_model.pth"

transform = transforms.Compose([
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

root = "../datasets/rgb2point_v4/metadata/"

train_dataset = RGB2PointV4Dataset(
    root + "procedural_train.json"
)
train_dataset.transform = transform

val_dataset = RGB2PointV4Dataset(
    root + "procedural_val.json"
)
val_dataset.transform = transform

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = RGB2PointV4(pretrained=True).to(device)

optimizer = torch.optim.AdamW(
    [
        p for p in model.parameters()
        if p.requires_grad
    ],
    lr=LEARNING_RATE
)

best_val_cd = float("inf")
best_epoch = 0
start = time.time()

print("EXPERIMENT:", args.name, flush=True)
print("DEVICE:", device, flush=True)
print("TRAIN_IMAGES:", len(train_dataset), flush=True)
print("VAL_IMAGES:", len(val_dataset), flush=True)
print("EPOCHS:", EPOCHS, flush=True)
print("SEED:", SEED, flush=True)
print("COVERAGE_WEIGHT:", COVERAGE_WEIGHT, flush=True)
print("DIMENSION_WEIGHT:", DIMENSION_WEIGHT, flush=True)

for epoch in range(EPOCHS):

    model.train()

    total_sum = 0.0
    coverage_sum = 0.0
    dimension_sum = 0.0

    for batch in train_loader:

        images = batch["image"].to(device)
        gt = batch["points"].to(device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(images)

        total, coverage, dimension = geometry_aware_loss(
            pred,
            gt,
            coverage_weight=COVERAGE_WEIGHT,
            dimension_weight=DIMENSION_WEIGHT
        )

        total.backward()
        optimizer.step()

        n = images.shape[0]

        total_sum += total.item() * n
        coverage_sum += coverage.item() * n
        dimension_sum += dimension.item() * n

    model.eval()

    val_cd_sum = 0.0
    val_dim_sum = 0.0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(device)
            gt = batch["points"].to(device)

            pred = model(images)

            distances = torch.cdist(pred, gt)

            p2g = distances.min(dim=2).values.mean(dim=1)
            g2p = distances.min(dim=1).values.mean(dim=1)

            cd = p2g + g2p

            pred_dims = pred.amax(dim=1) - pred.amin(dim=1)
            gt_dims = gt.amax(dim=1) - gt.amin(dim=1)

            dim_error = (
                (pred_dims - gt_dims).abs()
                / gt_dims.clamp_min(1e-6)
            ).mean(dim=1)

            val_cd_sum += cd.sum().item()
            val_dim_sum += dim_error.sum().item()

    val_cd = val_cd_sum / len(val_dataset)
    val_dim = val_dim_sum / len(val_dataset)

    marker = ""

    if val_cd < best_val_cd:

        best_val_cd = val_cd
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            checkpoint
        )

        marker = " *BEST*"

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Train: {total_sum / len(train_dataset):.6f} "
        f"| Coverage: {coverage_sum / len(train_dataset):.6f} "
        f"| Dimension: {dimension_sum / len(train_dataset):.6f} "
        f"| Val CD: {val_cd:.6f} "
        f"| Val Dim: {val_dim * 100:.2f}%"
        f"{marker}",
        flush=True
    )

print("\nABLATION_COMPLETE")
print("EXPERIMENT:", args.name)
print("DIMENSION_WEIGHT:", DIMENSION_WEIGHT)
print("BEST_EPOCH:", best_epoch)
print("BEST_VAL_CD:", round(best_val_cd, 6))
print("TRAINING_SECONDS:", round(time.time() - start, 2))
print("CHECKPOINT:", checkpoint)

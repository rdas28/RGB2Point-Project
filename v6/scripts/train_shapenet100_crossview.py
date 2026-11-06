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
from v4.scripts.dataset_v4 import RGB2PointV4Dataset
from v6.models.cross_view_loss import cross_view_loss
from v6.scripts.paired_view_dataset import PairedViewDataset


SEED = 42
EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
COVERAGE_WEIGHT = 2.0
DIMENSION_WEIGHT = 0.1
CONSISTENCY_WEIGHT = 0.1

CHECKPOINT_PATH = Path(
    "v6/outputs/shapenet100_crossview_best_model.pth"
)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--smoke-test", action="store_true")
args = parser.parse_args()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

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

root = "datasets/rgb2point_v4/metadata/"

train_dataset = PairedViewDataset(
    root + "shapenet100_train.json"
)

val_dataset = RGB2PointV4Dataset(
    root + "shapenet100_val.json"
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
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE
)

best_val_cd = float("inf")
start = time.time()

if not args.smoke_test:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("EXPERIMENT: SHAPENET100_CROSSVIEW_V6", flush=True)
print("DEVICE:", device, flush=True)
print("TRAIN_OBJECTS:", len(train_dataset.objects), flush=True)
print("TRAIN_PAIRS_PER_EPOCH:", len(train_dataset), flush=True)
print("TRAIN_IMAGES_PER_EPOCH:", len(train_dataset) * 2, flush=True)
print("VAL_IMAGES:", len(val_dataset), flush=True)
print("EPOCHS:", args.epochs, flush=True)
print("BATCH_SIZE_PAIRS:", BATCH_SIZE, flush=True)
print("LEARNING_RATE:", LEARNING_RATE, flush=True)
print("CONSISTENCY_WEIGHT:", CONSISTENCY_WEIGHT, flush=True)
print("SMOKE_TEST:", args.smoke_test, flush=True)

for epoch in range(args.epochs):
    model.train()

    total_loss_sum = 0.0
    geometry_loss_sum = 0.0
    consistency_loss_sum = 0.0
    processed_pairs = 0

    for batch_index, batch in enumerate(train_loader):
        image_a = batch["image_a"].to(device)
        image_b = batch["image_b"].to(device)
        gt = batch["points"].to(device)

        optimizer.zero_grad()

        images = torch.cat([image_a, image_b], dim=0)
        predictions = model(images)

        pred_a, pred_b = predictions.chunk(2, dim=0)

        total_loss, geometry_loss, consistency_loss = (
            cross_view_loss(
                pred_a,
                pred_b,
                gt,
                coverage_weight=COVERAGE_WEIGHT,
                dimension_weight=DIMENSION_WEIGHT,
                consistency_weight=CONSISTENCY_WEIGHT
            )
        )

        if not torch.isfinite(total_loss).item():
            raise RuntimeError("Non-finite V6 training loss")

        total_loss.backward()
        optimizer.step()

        pair_count = gt.shape[0]
        processed_pairs += pair_count

        total_loss_sum += total_loss.item() * pair_count
        geometry_loss_sum += geometry_loss.item() * pair_count
        consistency_loss_sum += consistency_loss.item() * pair_count

        if args.smoke_test and batch_index >= 1:
            break

    train_total = total_loss_sum / processed_pairs
    train_geometry = geometry_loss_sum / processed_pairs
    train_consistency = consistency_loss_sum / processed_pairs

    model.eval()

    val_cd_sum = 0.0
    val_dim_error_sum = 0.0
    val_samples = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
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
            val_dim_error_sum += dim_error.sum().item()
            val_samples += images.shape[0]

            if args.smoke_test and batch_index >= 1:
                break

    val_cd = val_cd_sum / val_samples
    val_dim_error = val_dim_error_sum / val_samples

    if val_cd < best_val_cd:
        best_val_cd = val_cd
        marker = " *BEST*"

        if not args.smoke_test:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
    else:
        marker = ""

    print(
        f"Epoch {epoch + 1:02d}/{args.epochs}"
        f" | Total Loss: {train_total:.6f}"
        f" | Geometry: {train_geometry:.6f}"
        f" | Consistency: {train_consistency:.6f}"
        f" | Val CD: {val_cd:.6f}"
        f" | Val Dim Error: {val_dim_error * 100:.2f}%"
        f"{marker}",
        flush=True
    )

print("V6_TRAINING_COMPLETE")
print("BEST_VAL_CD:", round(best_val_cd, 6))
print("TRAINING_SECONDS:", round(time.time() - start, 2))

if args.smoke_test:
    print("SMOKE_TEST_PASSED")
    print("CHECKPOINT_NOT_SAVED")
else:
    print("CHECKPOINT_SAVED:", CHECKPOINT_PATH)

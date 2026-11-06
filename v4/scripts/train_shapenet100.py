import time
import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from v4.models.rgb2point_v4 import RGB2PointV4
from v4.models.geometry_aware_loss import geometry_aware_loss
from v4.scripts.dataset_v4 import RGB2PointV4Dataset

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SEED = 42
EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 1e-4

COVERAGE_WEIGHT = 2.0
DIMENSION_WEIGHT = 0.1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------

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

# --------------------------------------------------
# DATASETS
# --------------------------------------------------

root = "../datasets/rgb2point_v4/metadata/"

train_dataset = RGB2PointV4Dataset(
    root + "shapenet100_train.json"
)
train_dataset.transform = transform

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

# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = RGB2PointV4(
    pretrained=True
).to(device)

optimizer = torch.optim.AdamW(
    [
        p for p in model.parameters()
        if p.requires_grad
    ],
    lr=LEARNING_RATE
)

best_val_cd = float("inf")

start = time.time()

print("EXPERIMENT: SHAPENET100", flush=True)
print("DEVICE:", device, flush=True)
print("TRAIN_IMAGES:", len(train_dataset), flush=True)
print("VAL_IMAGES:", len(val_dataset), flush=True)
print("EPOCHS:", EPOCHS, flush=True)
print("BATCH_SIZE:", BATCH_SIZE, flush=True)
print("COVERAGE_WEIGHT:", COVERAGE_WEIGHT, flush=True)
print("DIMENSION_WEIGHT:", DIMENSION_WEIGHT, flush=True)

# --------------------------------------------------
# TRAINING
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss_sum = 0.0
    coverage_loss_sum = 0.0
    dimension_loss_sum = 0.0

    for batch in train_loader:

        images = batch["image"].to(device)
        gt = batch["points"].to(device)

        optimizer.zero_grad()

        pred = model(images)

        total_loss, coverage_loss, dimension_loss = (
            geometry_aware_loss(
                pred,
                gt,
                coverage_weight=COVERAGE_WEIGHT,
                dimension_weight=DIMENSION_WEIGHT
            )
        )

        total_loss.backward()
        optimizer.step()

        batch_size = images.shape[0]

        total_loss_sum += total_loss.item() * batch_size
        coverage_loss_sum += coverage_loss.item() * batch_size
        dimension_loss_sum += dimension_loss.item() * batch_size

    train_total = total_loss_sum / len(train_dataset)
    train_coverage = coverage_loss_sum / len(train_dataset)
    train_dimension = dimension_loss_sum / len(train_dataset)

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    model.eval()

    val_cd_sum = 0.0
    val_dimension_error_sum = 0.0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(device)
            gt = batch["points"].to(device)

            pred = model(images)

            distances = torch.cdist(pred, gt)

            p2g = distances.min(dim=2).values.mean(dim=1)
            g2p = distances.min(dim=1).values.mean(dim=1)

            cd = p2g + g2p

            pred_dims = (
                pred.amax(dim=1) - pred.amin(dim=1)
            )

            gt_dims = (
                gt.amax(dim=1) - gt.amin(dim=1)
            )

            dimension_error = (
                (pred_dims - gt_dims).abs()
                / gt_dims.clamp_min(1e-6)
            ).mean(dim=1)

            batch_size = images.shape[0]

            val_cd_sum += cd.sum().item()
            val_dimension_error_sum += (
                dimension_error.sum().item()
            )

    val_cd = val_cd_sum / len(val_dataset)

    val_dimension_error = (
        val_dimension_error_sum / len(val_dataset)
    )

    # --------------------------------------------------
    # CHECKPOINT SELECTION
    # --------------------------------------------------

    if val_cd < best_val_cd:

        best_val_cd = val_cd

        torch.save(
            model.state_dict(),
            "v4/outputs/shapenet100_best_model.pth"
        )

        marker = " *BEST*"

    else:
        marker = ""

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Total Loss: {train_total:.6f} "
        f"| Coverage: {train_coverage:.6f} "
        f"| Dimension: {train_dimension:.6f} "
        f"| Val CD: {val_cd:.6f} "
        f"| Val Dim Error: {val_dimension_error * 100:.2f}%"
        f"{marker}",
        flush=True
    )

# --------------------------------------------------
# COMPLETION
# --------------------------------------------------

print("\nSHAPENET100_TRAINING_COMPLETE")
print("BEST_VAL_CD:", round(best_val_cd, 6))
print("TRAINING_SECONDS:", round(time.time() - start, 2))

print(
    "CHECKPOINT_SAVED:",
    "v4/outputs/shapenet100_best_model.pth"
)

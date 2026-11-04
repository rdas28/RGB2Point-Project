import time
import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from v4.models.rgb2point_v4 import RGB2PointV4
from v4.models.coverage_loss import coverage_chamfer_loss
from v4.scripts.dataset_v4 import RGB2PointV4Dataset

SEED = 42
EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 1e-4

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

root = "../datasets/rgb2point_v4/metadata/"

train_dataset = RGB2PointV4Dataset(
    root + "procedural_train.json"
)
train_dataset.transform = transform

val_dataset = RGB2PointV4Dataset(
    root + "procedural_val.json"
)
val_dataset.transform = transform

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = RGB2PointV4(
    pretrained=True
).to(device)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE
)

best_val_cd = float("inf")
start = time.time()

print("DEVICE:", device, flush=True)
print("TRAIN_IMAGES:", len(train_dataset), flush=True)
print("VAL_IMAGES:", len(val_dataset), flush=True)
print("EPOCHS:", EPOCHS, flush=True)

for epoch in range(EPOCHS):
    model.train()

    train_loss_sum = 0.0

    for batch in train_loader:
        images = batch["image"].to(device)
        gt = batch["points"].to(device)

        optimizer.zero_grad()

        pred = model(images)

        loss, _, _ = coverage_chamfer_loss(
            pred,
            gt,
            coverage_weight=2.0
        )

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * images.shape[0]

    train_loss = train_loss_sum / len(train_dataset)

    model.eval()
    val_cd_sum = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            gt = batch["points"].to(device)

            pred = model(images)

            distances = torch.cdist(pred, gt)

            p2g = distances.min(dim=2).values.mean(dim=1)
            g2p = distances.min(dim=1).values.mean(dim=1)

            cd = p2g + g2p

            val_cd_sum += cd.sum().item()

    val_cd = val_cd_sum / len(val_dataset)

    if val_cd < best_val_cd:
        best_val_cd = val_cd

        torch.save(
            model.state_dict(),
            "v4/outputs/multiobject_best_model.pth"
        )

        marker = " *BEST*"
    else:
        marker = ""

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Train Weighted Loss: {train_loss:.6f} "
        f"| Val CD: {val_cd:.6f}"
        f"{marker}",
        flush=True
    )

print("BEST_VAL_CD:", round(best_val_cd, 6))
print("TRAINING_SECONDS:", round(time.time() - start, 2))
print(
    "CHECKPOINT_SAVED: v4/outputs/multiobject_best_model.pth"
)

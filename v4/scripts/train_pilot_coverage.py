import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from v4.models.rgb2point_v4 import RGB2PointV4
from v4.scripts.dataset_v4 import RGB2PointV4Dataset
from v4.models.coverage_loss import coverage_chamfer_loss

SEED = 42
EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

dataset = RGB2PointV4Dataset(
    "../datasets/rgb2point_v4/metadata/pilot_manifest.json"
)

# Match the pretrained ViT's input normalization.
# The rendered images are already 224x224, so we do
# not apply an additional crop that could cut off the chair.
dataset.transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

model = RGB2PointV4(
    num_points=1024,
    hidden_dim=256,
    num_heads=4,
    num_decoder_layers=2,
    pretrained=True
).to(device)

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LEARNING_RATE
)

print("DEVICE:", device, flush=True)
print("DATASET_IMAGES:", len(dataset), flush=True)
print("UNIQUE_OBJECTS: 1", flush=True)
print("EPOCHS:", EPOCHS, flush=True)

start = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        gt = batch["points"].to(device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(images)
        loss, pred_to_gt, gt_to_pred = coverage_chamfer_loss(
            pred,
            gt,
            coverage_weight=2.0
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Train CD: {avg_loss:.6f}",
        flush=True
    )

print("TRAINING_SECONDS:", round(time.time() - start, 2))

checkpoint = "v4/outputs/pilot_coverage_model.pth"
torch.save(model.state_dict(), checkpoint)

print("CHECKPOINT_SAVED:", checkpoint)

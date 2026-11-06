from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from v4.models.rgb2point_v4 import RGB2PointV4
from v4.scripts.dataset_v4 import RGB2PointV4Dataset

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

dataset = RGB2PointV4Dataset(
    "datasets/rgb2point_v4/metadata/shapenet_test.json"
)

dataset.transform = transforms.Compose([
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

model = RGB2PointV4(pretrained=False).to(device)
state = torch.load(
    "v4/outputs/shapenet_pilot_best_model.pth",
    map_location=device,
    weights_only=True
)
model.load_state_dict(state)
model.eval()

indices = [0, 2, 4, 6]
fig = plt.figure(figsize=(16, 12))

for row, idx in enumerate(indices):
    sample = dataset[idx]
    image = sample["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image)[0].cpu().numpy()

    gt = sample["points"].numpy()

    rgb = image[0].cpu() * 0.5 + 0.5
    rgb = rgb.permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0, 1)

    ax_img = fig.add_subplot(4, 3, row * 3 + 1)
    ax_img.imshow(rgb)
    ax_img.set_title(f"RGB View {sample['view_id']}")
    ax_img.axis("off")

    combined = np.concatenate([gt, pred], axis=0)
    center = (combined.max(axis=0) + combined.min(axis=0)) / 2
    radius = max(
        float(np.max(combined.max(axis=0) - combined.min(axis=0))) / 2,
        0.01
    )

    for col, (points, title) in enumerate([
        (gt, "Ground Truth"),
        (pred, "Prediction")
    ], start=2):
        ax = fig.add_subplot(
            4, 3, row * 3 + col, projection="3d"
        )

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=1.5
        )

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=45)
        ax.set_title(title)

plt.tight_layout()

output = Path("v4/outputs/shapenet_pilot_predictions.png")
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=160)
plt.close()

print("VISUALIZATION_SAVED:", output)
print("SHAPENET_PILOT_VISUALIZATION_COMPLETE")

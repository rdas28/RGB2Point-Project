import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from v4.models.rgb2point_v4 import RGB2PointV4
from v4.scripts.dataset_v4 import RGB2PointV4Dataset

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

dataset = RGB2PointV4Dataset(
    "../datasets/rgb2point_v4/metadata/pilot_manifest.json"
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

sample = dataset[0]
image = sample["image"].unsqueeze(0).to(device)
gt = sample["points"].numpy()

def predict(checkpoint):
    model = RGB2PointV4(pretrained=False).to(device)
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    with torch.no_grad():
        result = model(image)[0].cpu().numpy()

    del model
    return result

baseline = predict("v4/outputs/pilot_100_model.pth")
weighted = predict("v4/outputs/pilot_coverage_model.pth")

rgb = (
    sample["image"].permute(1, 2, 0).numpy()
    * 0.5 + 0.5
).clip(0, 1)

fig = plt.figure(figsize=(17, 5))

ax = fig.add_subplot(141)
ax.imshow(rgb)
ax.set_title("Input RGB")
ax.axis("off")

clouds = [
    (baseline, "Baseline"),
    (weighted, "Coverage Weighted"),
    (gt, "Ground Truth")
]

all_points = np.vstack([baseline, weighted, gt])
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
center = (mins + maxs) / 2
radius = max(maxs - mins) * 0.55

for index, (points, title) in enumerate(clouds, start=2):
    ax = fig.add_subplot(1, 4, index, projection="3d")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=3,
        c=points[:, 2],
        cmap="viridis"
    )

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-60)
    ax.set_title(title)

plt.tight_layout()

output = "v4/outputs/loss_comparison.png"
plt.savefig(output, dpi=180)
plt.close(fig)

print("COMPARISON_SAVED:", output)

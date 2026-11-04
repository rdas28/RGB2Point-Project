import numpy as np
import torch
import matplotlib.pyplot as plt

from v4.models.rgb2point_v4 import RGB2PointV4
from v4.scripts.dataset_v4 import RGB2PointV4Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

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
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

model = RGB2PointV4(
    num_points=1024,
    hidden_dim=256,
    num_heads=4,
    num_decoder_layers=2,
    pretrained=False
).to(device)

model.load_state_dict(
    torch.load(
        "v4/outputs/pilot_best_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

sample = dataset[0]
image = sample["image"].unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(image)[0].cpu().numpy()

ground_truth = sample["points"].numpy()

rgb = (
    sample["image"].permute(1, 2, 0).numpy()
    * 0.5 + 0.5
).clip(0, 1)

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(131)
ax1.imshow(rgb)
ax1.set_title("Input RGB")
ax1.axis("off")

all_points = np.vstack([prediction, ground_truth])
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
center = (mins + maxs) / 2
radius = max(maxs - mins) / 2 * 1.1

for index, (points, title) in enumerate([
    (prediction, "V4 Prediction"),
    (ground_truth, "Ground Truth")
], start=2):
    ax = fig.add_subplot(1, 3, index, projection="3d")

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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

plt.tight_layout()

output = "v4/outputs/pilot_reconstruction.png"
plt.savefig(output, dpi=180)
plt.close(fig)

print("VISUALIZATION_SAVED:", output)
print("PREDICTION_MIN:", prediction.min(axis=0))
print("PREDICTION_MAX:", prediction.max(axis=0))
print("GROUND_TRUTH_MIN:", ground_truth.min(axis=0))
print("GROUND_TRUTH_MAX:", ground_truth.max(axis=0))

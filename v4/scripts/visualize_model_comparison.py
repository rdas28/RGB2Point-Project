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
    "../datasets/rgb2point_v4/metadata/procedural_val.json"
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

checkpoints = {
    "Baseline": "v4/outputs/multiobject_best_model.pth",
    "Geometry": "v4/outputs/multiobject_geometry_best_model.pth"
}

models = {}

for name, path in checkpoints.items():

    model = RGB2PointV4(pretrained=False).to(device)

    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()
    models[name] = model

selected = {}
for i in range(len(dataset)):
    category = dataset[i]["category"]
    if category not in selected:
        selected[category] = i

fig = plt.figure(figsize=(17, 9))

for row, category in enumerate(["chair", "table"]):

    sample = dataset[selected[category]]

    image = sample["image"].unsqueeze(0).to(device)
    gt = sample["points"].numpy()

    predictions = {}

    with torch.no_grad():
        for name, model in models.items():
            predictions[name] = (
                model(image)[0].cpu().numpy()
            )

    rgb = (
        sample["image"].permute(1, 2, 0).numpy()
        * 0.5 + 0.5
    ).clip(0, 1)

    clouds = [
        ("Baseline", predictions["Baseline"]),
        ("Geometry-aware", predictions["Geometry"]),
        ("Ground Truth", gt)
    ]

    combined = np.concatenate(
        [gt] + [p for p in predictions.values()],
        axis=0
    )

    low = combined.min(axis=0)
    high = combined.max(axis=0)
    center = (low + high) / 2
    radius = max((high - low).max() / 2, 1e-6)

    ax = fig.add_subplot(2, 4, row * 4 + 1)
    ax.imshow(rgb)
    ax.set_title(f"{category.upper()} - Input")
    ax.axis("off")

    for col, (name, points) in enumerate(clouds, start=2):

        ax = fig.add_subplot(
            2, 4, row * 4 + col,
            projection="3d"
        )

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=2,
            alpha=0.8
        )

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-60)
        ax.set_title(name)
        ax.set_axis_off()

plt.tight_layout()

output = "v4/outputs/baseline_vs_geometry_visual.png"

plt.savefig(output, dpi=180, bbox_inches="tight")
plt.close(fig)

print("COMPARISON_SAVED:", output)
print("OBJECTS:", {
    category: dataset[index]["object_id"]
    for category, index in selected.items()
})

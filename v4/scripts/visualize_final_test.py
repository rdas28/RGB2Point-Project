import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from v4.models.rgb2point_v4 import RGB2PointV4
from v4.scripts.dataset_v4 import RGB2PointV4Dataset

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

dataset = RGB2PointV4Dataset(
    "../datasets/rgb2point_v4/metadata/procedural_test.json"
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

model.load_state_dict(
    torch.load(
        "v4/outputs/multiobject_best_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

selected = {}

for i in range(len(dataset)):
    category = dataset[i]["category"]

    if category not in selected:
        selected[category] = i

fig = plt.figure(figsize=(14, 9))

for row, category in enumerate(["chair", "table"]):

    sample = dataset[selected[category]]

    image = sample["image"].unsqueeze(0).to(device)
    gt = sample["points"].numpy()

    with torch.inference_mode():
        pred = model(image)[0].cpu().numpy()

    distances = torch.cdist(
        torch.from_numpy(pred).unsqueeze(0),
        torch.from_numpy(gt).unsqueeze(0)
    )

    cd = (
        distances.min(dim=2).values.mean()
        + distances.min(dim=1).values.mean()
    ).item()

    rgb = (
        sample["image"].permute(1, 2, 0).numpy()
        * 0.5 + 0.5
    ).clip(0, 1)

    combined = np.concatenate([pred, gt], axis=0)

    low = combined.min(axis=0)
    high = combined.max(axis=0)

    center = (low + high) / 2
    radius = max((high - low).max() / 2, 1e-6)

    ax = fig.add_subplot(2, 3, row * 3 + 1)

    ax.imshow(rgb)
    ax.set_title(
        f"{category.upper()} — RGB Input",
        fontsize=13,
        fontweight="bold"
    )
    ax.axis("off")

    for col, (name, points) in enumerate([
        ("Predicted Point Cloud", pred),
        ("Ground Truth", gt)
    ], start=2):

        ax = fig.add_subplot(
            2, 3, row * 3 + col,
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

        title = name

        if col == 2:
            title += f"\nChamfer: {cd:.5f}"

        ax.set_title(title, fontsize=12)
        ax.set_axis_off()

    print(
        f"{category.upper()} | "
        f"OBJECT: {sample['object_id']} | "
        f"CHAMFER: {cd:.6f}",
        flush=True
    )

fig.suptitle(
    "RGB2Point v4 — Held-Out Test Reconstructions",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

output = "v4/outputs/final_test_reconstructions.png"

plt.savefig(
    output,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nFINAL_VISUALIZATION_SAVED:", output)
print("DEVICE:", device)

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

model = RGB2PointV4(pretrained=False).to(device)

model.load_state_dict(
    torch.load(
        "v4/outputs/multiobject_best_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

# Select one held-out chair and one held-out table.
indices = []
seen_categories = set()

for i in range(len(dataset)):
    category = dataset[i]["category"]

    if category not in seen_categories:
        indices.append(i)
        seen_categories.add(category)

    if len(indices) == 2:
        break

fig = plt.figure(figsize=(13, 9))

for row, index in enumerate(indices):
    sample = dataset[index]

    image = sample["image"].unsqueeze(0).to(device)
    gt = sample["points"].cpu().numpy()

    with torch.no_grad():
        pred = model(image)[0].cpu().numpy()

    rgb = (
        sample["image"].permute(1, 2, 0).numpy()
        * 0.5 + 0.5
    ).clip(0, 1)

    ax = fig.add_subplot(2, 3, row * 3 + 1)
    ax.imshow(rgb)
    ax.set_title(
        f'{sample["object_id"]}\nInput RGB'
    )
    ax.axis("off")

    combined = np.vstack([pred, gt])
    center = (combined.min(axis=0) + combined.max(axis=0)) / 2
    radius = np.max(combined.max(axis=0) - combined.min(axis=0)) * 0.55

    for col, (points, title) in enumerate([
        (pred, "Predicted"),
        (gt, "Ground Truth")
    ], start=2):

        ax = fig.add_subplot(
            2, 3, row * 3 + col,
            projection="3d"
        )

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

output = "v4/outputs/multiobject_val_reconstruction.png"

plt.savefig(output, dpi=180)
plt.close(fig)

print("VISUALIZATION_SAVED:", output)
print("OBJECTS_VISUALIZED:", [
    dataset[i]["object_id"] for i in indices
])

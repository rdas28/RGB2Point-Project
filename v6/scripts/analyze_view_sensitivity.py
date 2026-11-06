import json
from collections import defaultdict
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from v4.models.rgb2point_v4 import RGB2PointV4
from v4.scripts.dataset_v4 import RGB2PointV4Dataset

torch.manual_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

manifest = Path(
    "datasets/rgb2point_v4/metadata/shapenet100_test.json"
)

dataset = RGB2PointV4Dataset(str(manifest))

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

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

model = RGB2PointV4(pretrained=False).to(device)

checkpoint = torch.load(
    "v6/outputs/shapenet100_crossview_best_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

predictions = defaultdict(dict)
ground_truth = {}

with torch.no_grad():
    index = 0

    for batch in loader:
        images = batch["image"].to(device)
        pred = model(images).cpu()
        gt = batch["points"].cpu()

        for j in range(images.shape[0]):
            record = dataset.records[index]
            object_id = record["object_id"]
            view_id = int(record["view_id"])

            predictions[object_id][view_id] = pred[j]
            ground_truth[object_id] = gt[j]

            index += 1

assert index == 120
assert len(predictions) == 15
assert all(len(views) == 8 for views in predictions.values())

def chamfer(a, b):
    distances = torch.cdist(
        a.unsqueeze(0),
        b.unsqueeze(0)
    )[0]

    return (
        distances.min(dim=1).values.mean()
        + distances.min(dim=0).values.mean()
    ).item()

same_chair = []
different_chair = []
different_gt = []

object_ids = sorted(predictions)

for object_id in object_ids:
    views = predictions[object_id]

    for i in range(8):
        for j in range(i + 1, 8):
            same_chair.append(
                chamfer(views[i], views[j])
            )

for i in range(len(object_ids)):
    for j in range(i + 1, len(object_ids)):
        object_a = object_ids[i]
        object_b = object_ids[j]

        different_chair.append(
            chamfer(
                predictions[object_a][0],
                predictions[object_b][0]
            )
        )

        different_gt.append(
            chamfer(
                ground_truth[object_a],
                ground_truth[object_b]
            )
        )

mean_same = sum(same_chair) / len(same_chair)
mean_different = sum(different_chair) / len(different_chair)
mean_gt = sum(different_gt) / len(different_gt)

print("DEVICE:", device)
print("TEST_OBJECTS:", len(object_ids))
print("TEST_IMAGES:", index)
print("SAME_CHAIR_PAIRS:", len(same_chair))
print("DIFFERENT_CHAIR_PAIRS:", len(different_chair))
print("MEAN_SAME_CHAIR_CD:", round(mean_same, 6))
print("MEAN_DIFFERENT_CHAIR_CD:", round(mean_different, 6))
print("MEAN_DIFFERENT_GT_CD:", round(mean_gt, 6))
print(
    "BETWEEN_TO_WITHIN_RATIO:",
    round(mean_different / max(mean_same, 1e-8), 4)
)
print("V6_VIEW_SENSITIVITY_ANALYSIS_COMPLETE")

import torch
from collections import defaultdict
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from v4.models.rgb2point_v4 import RGB2PointV4
from v4.scripts.dataset_v4 import RGB2PointV4Dataset

SEED = 42
torch.manual_seed(SEED)

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

results = defaultdict(lambda: defaultdict(list))
object_ids = set()

with torch.inference_mode():
    for sample in dataset:
        image = sample["image"].unsqueeze(0).to(device)
        gt = sample["points"].unsqueeze(0).to(device)

        pred = model(image)

        distances = torch.cdist(pred, gt)
        p2g = distances.min(dim=2).values
        g2p = distances.min(dim=1).values

        cd = p2g.mean() + g2p.mean()

        pred_dims = pred.amax(dim=1) - pred.amin(dim=1)
        gt_dims = gt.amax(dim=1) - gt.amin(dim=1)

        dim_error = (
            (pred_dims - gt_dims).abs()
            / gt_dims.clamp_min(1e-6)
        ).mean()

        category = sample["category"]
        object_ids.add(sample["object_id"])

        metrics = {
            "cd": cd.item(),
            "p2g": p2g.mean().item(),
            "g2p": g2p.mean().item(),
            "coverage_002": (g2p < 0.02).float().mean().item(),
            "coverage_005": (g2p < 0.05).float().mean().item(),
            "coverage_010": (g2p < 0.10).float().mean().item(),
            "dim_error": dim_error.item()
        }

        for key, value in metrics.items():
            results[category][key].append(value)
            results["OVERALL"][key].append(value)

print("\nFINAL_HELD_OUT_TEST")
print("DEVICE:", device)
print("MODEL: COVERAGE_WEIGHTED_BASELINE")
print("TEST_IMAGES:", len(dataset))
print("UNSEEN_OBJECTS:", len(object_ids))

for category in sorted(results):
    metrics = results[category]

    def avg(key):
        return sum(metrics[key]) / len(metrics[key])

    print("\nCATEGORY:", category.upper())
    print("IMAGES:", len(metrics["cd"]))
    print("MEAN_CHAMFER:", round(avg("cd"), 6))
    print("PRED_TO_GT:", round(avg("p2g"), 6))
    print("GT_TO_PRED:", round(avg("g2p"), 6))
    print("COVERAGE_0.02:", f'{avg("coverage_002") * 100:.2f}%')
    print("COVERAGE_0.05:", f'{avg("coverage_005") * 100:.2f}%')
    print("COVERAGE_0.10:", f'{avg("coverage_010") * 100:.2f}%')
    print("DIMENSION_ERROR:", f'{avg("dim_error") * 100:.2f}%')

print("\nFINAL_TEST_EVALUATION_COMPLETE")

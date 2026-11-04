import torch
from collections import defaultdict
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

results = defaultdict(lambda: defaultdict(list))

with torch.no_grad():
    for sample in dataset:
        image = sample["image"].unsqueeze(0).to(device)
        gt = sample["points"].unsqueeze(0).to(device)

        pred = model(image)

        distances = torch.cdist(pred, gt)

        p2g = distances.min(dim=2).values
        g2p = distances.min(dim=1).values

        cd = p2g.mean() + g2p.mean()

        category = sample["category"]

        results[category]["cd"].append(cd.item())
        results[category]["coverage_002"].append(
            (g2p < 0.02).float().mean().item()
        )
        results[category]["coverage_005"].append(
            (g2p < 0.05).float().mean().item()
        )

        pred_dims = pred.amax(dim=1) - pred.amin(dim=1)
        gt_dims = gt.amax(dim=1) - gt.amin(dim=1)

        dim_error = (
            (pred_dims - gt_dims).abs() / gt_dims
        ).mean().item()

        results[category]["dim_error"].append(dim_error)

for category, metrics in sorted(results.items()):
    def avg(key):
        values = metrics[key]
        return sum(values) / len(values)

    print(f"\nCATEGORY: {category.upper()}")
    print("IMAGES:", len(metrics["cd"]))
    print("MEAN_CHAMFER:", round(avg("cd"), 6))
    print("COVERAGE_0.02:", f'{avg("coverage_002") * 100:.2f}%')
    print("COVERAGE_0.05:", f'{avg("coverage_005") * 100:.2f}%')
    print("MEAN_DIMENSION_ERROR:", f'{avg("dim_error") * 100:.2f}%')

print("\nVALIDATION_EVALUATION_COMPLETE")

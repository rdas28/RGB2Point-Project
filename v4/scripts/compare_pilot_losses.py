import torch
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

checkpoints = {
    "BASELINE": "v4/outputs/pilot_100_model.pth",
    "COVERAGE_WEIGHTED": "v4/outputs/pilot_coverage_model.pth"
}

for name, checkpoint in checkpoints.items():
    model = RGB2PointV4(pretrained=False).to(device)

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    results = {
        "cd": [],
        "pred_to_gt": [],
        "gt_to_pred": [],
        "coverage_002": [],
        "coverage_005": [],
        "coverage_010": [],
        "dimension_error": []
    }

    with torch.no_grad():
        for sample in dataset:
            image = sample["image"].unsqueeze(0).to(device)
            gt = sample["points"].unsqueeze(0).to(device)

            pred = model(image)

            distances = torch.cdist(pred, gt)

            p2g = distances.min(dim=2).values
            g2p = distances.min(dim=1).values

            results["pred_to_gt"].append(p2g.mean().item())
            results["gt_to_pred"].append(g2p.mean().item())

            results["cd"].append(
                (p2g.mean() + g2p.mean()).item()
            )

            for threshold, key in [
                (0.02, "coverage_002"),
                (0.05, "coverage_005"),
                (0.10, "coverage_010")
            ]:
                results[key].append(
                    (g2p < threshold).float().mean().item()
                )

            pred_dims = pred.amax(dim=1) - pred.amin(dim=1)
            gt_dims = gt.amax(dim=1) - gt.amin(dim=1)

            dimension_error = (
                ((pred_dims - gt_dims).abs() / gt_dims)
                .mean()
                .item()
            )

            results["dimension_error"].append(dimension_error)

    def avg(key):
        return sum(results[key]) / len(results[key])

    print(f"\nMODEL: {name}")
    print("MEAN_CHAMFER:", round(avg("cd"), 6))
    print("MEAN_PRED_TO_GT:", round(avg("pred_to_gt"), 6))
    print("MEAN_GT_TO_PRED:", round(avg("gt_to_pred"), 6))
    print("COVERAGE_0.02:", f'{avg("coverage_002") * 100:.2f}%')
    print("COVERAGE_0.05:", f'{avg("coverage_005") * 100:.2f}%')
    print("COVERAGE_0.10:", f'{avg("coverage_010") * 100:.2f}%')
    print(
        "MEAN_DIMENSION_ERROR:",
        f'{avg("dimension_error") * 100:.2f}%'
    )

print("\nCOMPARISON_COMPLETE")

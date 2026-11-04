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

checkpoints = {
    "BASELINE": "v4/outputs/multiobject_best_model.pth",
    "GEOMETRY_AWARE": "v4/outputs/multiobject_geometry_best_model.pth"
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

    results = defaultdict(list)

    with torch.no_grad():

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

            results[(category, "cd")].append(cd.item())
            results[(category, "p2g")].append(p2g.mean().item())
            results[(category, "g2p")].append(g2p.mean().item())
            results[(category, "dim")].append(dim_error.item())

            for threshold in [0.02, 0.05, 0.10]:
                coverage = (
                    g2p < threshold
                ).float().mean().item()

                results[(category, f"coverage_{threshold}")].append(
                    coverage
                )

    print(f"\n{'=' * 45}")
    print("MODEL:", name)
    print(f"{'=' * 45}")

    for category in ["chair", "table"]:

        print(f"\nCATEGORY: {category.upper()}")

        for metric in [
            "cd", "p2g", "g2p",
            "coverage_0.02",
            "coverage_0.05",
            "coverage_0.1",
            "dim"
        ]:

            values = results[(category, metric)]
            average = sum(values) / len(values)

            if "coverage" in metric:
                average *= 100

            if metric == "dim":
                average *= 100

            print(
                f"{metric}: {average:.6f}"
            )

print("\nMODEL_COMPARISON_COMPLETE")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from v4.models.rgb2point_v4 import RGB2PointV4
from v4.scripts.dataset_v4 import RGB2PointV4Dataset

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

dataset = RGB2PointV4Dataset(
    "datasets/rgb2point_v4/metadata/shapenet100_test.json"
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

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

model = RGB2PointV4(pretrained=False).to(device)

checkpoint = torch.load(
    "v5/outputs/shapenet100_surface_best_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

total_cd = 0.0
total_dim_error = 0.0
total_samples = 0

with torch.no_grad():
    for batch in loader:
        images = batch["image"].to(device)
        gt = batch["points"].to(device)

        pred = model(images)

        distances = torch.cdist(pred, gt)

        p2g = distances.min(dim=2).values.mean(dim=1)
        g2p = distances.min(dim=1).values.mean(dim=1)
        cd = p2g + g2p

        pred_dims = pred.amax(dim=1) - pred.amin(dim=1)
        gt_dims = gt.amax(dim=1) - gt.amin(dim=1)

        dim_error = (
            (pred_dims - gt_dims).abs()
            / gt_dims.clamp_min(1e-6)
        ).mean(dim=1)

        total_cd += cd.sum().item()
        total_dim_error += dim_error.sum().item()
        total_samples += images.shape[0]

print("\nSHAPENET100_SURFACE_V5_TEST_RESULTS")
print("TEST_IMAGES:", total_samples)
print("TEST_OBJECTS:", len(set(dataset.records[i]["object_id"] for i in range(len(dataset)))))
print("MEAN_TEST_CD:", round(total_cd / total_samples, 6))
print("MEAN_DIMENSION_ERROR_PERCENT:", round(100 * total_dim_error / total_samples, 2))
print("SHAPENET_TEST_COMPLETE")

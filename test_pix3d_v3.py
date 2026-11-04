
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.rgb2point import RGB2Point
from data_loaders.pix3d_dataset import Pix3DDataset
from utils.chamfer_distance import chamfer_distance

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = Pix3DDataset(
    images_root="datasets/pix3d/images/",
    annotations_file="pix3d_test.json",
    points_root="datasets/pix3d/points/",
    transform=transform,
    num_points=1024,
    category="wardrobe"
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = RGB2Point(pretrained=False).to(device)
model.load_state_dict(
    torch.load(
        "outputs/pix3d_v3_best_model.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()

losses = []
times = []

with torch.inference_mode():
    for idx, (image, gt) in enumerate(loader):
        image = image.to(device)
        gt = gt.to(device).float()

        start = time.perf_counter()
        pred = model(image)
        elapsed = time.perf_counter() - start

        cd = chamfer_distance(pred, gt).item()
        losses.append(cd)
        times.append(elapsed)

        if (idx + 1) % 10 == 0:
            print(f"Evaluated {idx+1}/{len(dataset)} images")

print("\n===== HELD-OUT PIX3D TEST RESULTS =====")
print("Test images:", len(losses))
print("Mean Chamfer Distance:", sum(losses) / len(losses))
print("Minimum Chamfer Distance:", min(losses))
print("Maximum Chamfer Distance:", max(losses))
print("Mean CPU inference time:", sum(times) / len(times), "seconds")


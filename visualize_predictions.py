import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.rgb2point import RGB2Point
from datasets.pix3d_dataset import Pix3DDataset
from utils.visualize_pointcloud import visualize_input_output
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = RGB2Point().to(device)
model.load_state_dict(torch.load('outputs/best_model.pth'))
model.eval()

# Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

pix3d_dataset = Pix3DDataset(
    images_root="datasets/pix3d/images/",
    annotations_file="pix3d.json",  # ✅ Fixed path
    points_root="datasets/pix3d/points/",
    transform=transform,
    num_points=1024,
    category='chair'  # ✅ Optional, but cleaner to load only chairs
)

pix3d_loader = DataLoader(pix3d_dataset, batch_size=1, shuffle=False)

os.makedirs('outputs/predictions/', exist_ok=True)

for idx, (image, _) in enumerate(pix3d_loader):
    image = image.to(device)

    with torch.no_grad():
        pred_points = model(image).squeeze(0)

    visualize_input_output(image.squeeze(0), pred_points, f'outputs/predictions/vis_{idx}.png')

    if idx == 4:  # Save 5 visualizations
        break

print("✅ Visualization done! 5 images saved!")

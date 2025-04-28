import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.rgb2point import RGB2Point
from datasets.pix3d_dataset import Pix3DDataset
from utils.visualize_pointcloud import save_pointcloud, visualize_input_output
import os
import matplotlib.pyplot as plt
import imageio

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = RGB2Point().to(device)
model.load_state_dict(torch.load('outputs/best_model.pth'))
model.eval()

print("✅ eval.py is running correctly!")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Pix3D dataset
pix3d_dataset = Pix3DDataset(
    images_root="datasets/pix3d/images/",
    annotations_file="datasets/pix3d/pix3d.json",
    points_root="datasets/pix3d/points/",
    transform=transform,
    num_points=1024,
    category='wardrobe'
)

# 🔥 Add debug: How many samples loaded
print(f"✅ Number of samples loaded from Pix3D: {len(pix3d_dataset)}")

# Setup dataloader
pix3d_loader = DataLoader(pix3d_dataset, batch_size=1, shuffle=False)

# Ensure output folder exists
os.makedirs('outputs/predictions/', exist_ok=True)

# Predict and save
for idx, (image, _) in enumerate(pix3d_loader):
    image = image.to(device)

    with torch.no_grad():
        pred_points = model(image).squeeze(0)  # shape [1024, 3]

    # 🔥 Debug point cloud shape
    print(f"[DEBUG] Sample {idx}: pred_points shape {pred_points.shape}")

    # Save point cloud to .ply
    save_pointcloud(pred_points, f'outputs/predictions/pred_{idx}.ply')

    # Create a static image of the point cloud
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    pred_np = pred_points.detach().cpu().numpy()
    ax.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], s=1)
    ax.set_title(f'Predicted 3D Point Cloud {idx}')
    ax.axis('off')

    # Save the static image as .png
    plt.tight_layout()
    plt.savefig(f'outputs/predictions/static_pred_{idx}.png', dpi=300)
    plt.close()

    # Create a rotating GIF of the point cloud
    images = []
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], s=1)
    ax.set_title(f'Predicted 3D Point Cloud {idx}')
    ax.axis('off')

    # Rotate and save frames for the GIF
    for angle in range(0, 360, 10):
        ax.view_init(azim=angle)
        plt.draw()
        # Save the frame into the list
        img_path = f'outputs/predictions/frame_{angle}.png'
        plt.savefig(img_path)
        images.append(imageio.imread(img_path))
        os.remove(img_path)  # Clean up the individual frame images

    # Save the GIF
    gif_path = f'outputs/predictions/rotate_pred_{idx}.gif'
    imageio.mimsave(gif_path, images, duration=0.1)

    if idx == 4:  # Only first 5 samples
        break

print("✅ Evaluation done! 5 real samples saved in outputs/predictions/")

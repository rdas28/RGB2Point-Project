import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from v4.models.rgb2point_v4 import RGB2PointV4


def save_ply(points, output_path):
    """Save an (N, 3) array as an ASCII PLY point cloud."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="RGB2Point v4: predict a 3D point cloud from an RGB image."
    )
    parser.add_argument("--image", required=True, help="Input RGB image")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument(
        "--output",
        default="v4/outputs/prediction.ply",
        help="Output PLY file",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save XYZ coordinates as a NumPy array",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")

    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ])

    with Image.open(image_path) as img:
        image = transform(img.convert("RGB")).unsqueeze(0).to(device)

    model = RGB2PointV4(pretrained=False).to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    with torch.inference_mode():
        points = model(image)[0].cpu().numpy()

    if points.shape != (1024, 3):
        raise RuntimeError(f"Unexpected output shape: {points.shape}")

    save_ply(points, output_path)
    print(f"Saved {len(points)} points to: {output_path}")

    if args.save_npy:
        npy_path = output_path.with_suffix(".npy")
        np.save(npy_path, points)
        print(f"Saved NumPy array to: {npy_path}")


if __name__ == "__main__":
    main()

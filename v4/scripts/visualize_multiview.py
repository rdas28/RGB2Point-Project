from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

image_dir = Path("../datasets/rgb2point_v4/images")
output = Path("v4/outputs/test_chair_multiview.png")

fig, axes = plt.subplots(2, 4, figsize=(12, 7))

for i, ax in enumerate(axes.flat):
    image_path = image_dir / f"test_chair_view_{i:02d}.png"

    with Image.open(image_path) as img:
        ax.imshow(img.convert("RGB"))

    ax.set_title(f"View {i:02d} ({i * 45}°)")
    ax.axis("off")

fig.suptitle("RGB2Point v4 — Chair Multi-View Dataset", fontsize=15)
plt.tight_layout()
plt.savefig(output, dpi=160, bbox_inches="tight")
plt.close(fig)

print("CONTACT_SHEET_SAVED:", output)

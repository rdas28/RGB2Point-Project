import json
from pathlib import Path

root = Path("../datasets/rgb2point_v4").resolve()

records = []

for view_id in range(8):
    image = root / "images" / f"test_chair_view_{view_id:02d}.png"
    points = root / "points" / "test_chair.npy"

    if not image.is_file() or not points.is_file():
        raise FileNotFoundError(f"Missing image or point cloud: {image}")

    records.append({
        "object_id": "test_chair",
        "category": "chair",
        "view_id": view_id,
        "azimuth_degrees": view_id * 45,
        "image": str(image.relative_to(root)),
        "points": str(points.relative_to(root)),
        "coordinate_system": "Z-up"
    })

output = root / "metadata" / "pilot_manifest.json"

with output.open("w") as f:
    json.dump(records, f, indent=2)

print("MANIFEST_SAVED:", output)
print("TOTAL_IMAGES:", len(records))
print("UNIQUE_OBJECTS:", len(set(r["object_id"] for r in records)))
print("FIRST_RECORD:", json.dumps(records[0], indent=2))

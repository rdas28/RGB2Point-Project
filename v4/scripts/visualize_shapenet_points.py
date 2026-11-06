import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("../datasets/rgb2point_v4").resolve()
META = ROOT / "metadata" / "shapenet_all.json"

with META.open() as f:
    records = json.load(f)

objects = sorted(set(r["object_id"] for r in records))

fig = plt.figure(figsize=(16, 10))

for i, object_id in enumerate(objects):
    record = next(
        r for r in records
        if r["object_id"] == object_id
    )

    points = np.load(ROOT / record["points"])

    assert points.shape == (1024, 3)
    assert np.isfinite(points).all()

    print(
        object_id,
        "Shape:", points.shape,
        "Min:", points.min(axis=0),
        "Max:", points.max(axis=0)
    )

    ax = fig.add_subplot(2, 3, i + 1, projection="3d")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1,
        alpha=0.7
    )

    ax.set_title(object_id[:20], fontsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    center = (points.max(axis=0) + points.min(axis=0)) / 2
    radius = (points.max(axis=0) - points.min(axis=0)).max() / 2

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.view_init(elev=20, azim=45)

plt.tight_layout()

output = Path("v4/outputs/shapenet_pointclouds.png")
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=180)
print("SAVED:", output)
print("POINT_CLOUD_VALIDATION_PASSED")

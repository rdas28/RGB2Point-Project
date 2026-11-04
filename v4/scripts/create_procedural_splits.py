import json
import random
from pathlib import Path

ROOT = Path("../datasets/rgb2point_v4").resolve()
META_DIR = ROOT / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(42)

splits = {
    "train": [],
    "val": [],
    "test": []
}

for category in ["chair", "table"]:
    objects = [f"{category}_{i:03d}" for i in range(10)]
    rng.shuffle(objects)

    assignments = {
        "train": objects[:7],
        "val": objects[7:9],
        "test": objects[9:]
    }

    for split, object_ids in assignments.items():
        for object_id in object_ids:
            point_path = f"points/procedural/{object_id}.npy"

            assert (ROOT / point_path).is_file(), point_path

            for view_id in range(8):
                image_path = (
                    f"images/procedural/"
                    f"{object_id}_view_{view_id:02d}.png"
                )

                assert (ROOT / image_path).is_file(), image_path

                splits[split].append({
                    "object_id": object_id,
                    "category": category,
                    "view_id": view_id,
                    "azimuth_degrees": view_id * 45,
                    "image": image_path,
                    "points": point_path,
                    "coordinate_system": "Z-up"
                })

for split, records in splits.items():
    output = META_DIR / f"procedural_{split}.json"

    with open(output, "w") as f:
        json.dump(records, f, indent=2)

    object_ids = {r["object_id"] for r in records}

    print(
        split.upper(),
        "OBJECTS:", len(object_ids),
        "IMAGES:", len(records)
    )

object_sets = [
    {r["object_id"] for r in splits[name]}
    for name in ["train", "val", "test"]
]

assert object_sets[0].isdisjoint(object_sets[1])
assert object_sets[0].isdisjoint(object_sets[2])
assert object_sets[1].isdisjoint(object_sets[2])

print("OBJECT_LEAKAGE: NONE")
print("SPLITS_CREATED_SUCCESSFULLY")

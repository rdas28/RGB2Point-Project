import json
import random
from pathlib import Path
from collections import Counter

ROOT = Path("../datasets/rgb2point_v4").resolve()
META = ROOT / "metadata"

with (META / "shapenet_all.json").open() as f:
    records = json.load(f)

objects = sorted(set(r["object_id"] for r in records))

rng = random.Random(42)
rng.shuffle(objects)

train_ids = set(objects[:3])
val_ids = set(objects[3:4])
test_ids = set(objects[4:])

splits = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
}

assert not (train_ids & val_ids)
assert not (train_ids & test_ids)
assert not (val_ids & test_ids)

for split_name, object_ids in splits.items():
    subset = [
        r for r in records
        if r["object_id"] in object_ids
    ]

    output = META / f"shapenet_{split_name}.json"

    with output.open("w") as f:
        json.dump(subset, f, indent=2)

    print(
        split_name.upper(),
        "Objects:", len(object_ids),
        "Images:", len(subset),
        "Categories:", dict(Counter(r["category"] for r in subset))
    )

print("OBJECT_DISJOINT_SPLITS_CREATED")

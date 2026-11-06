import json
import random
from collections import Counter
from pathlib import Path

SEED = 42

root = Path("../datasets/rgb2point_v4/metadata")
records = json.loads((root / "shapenet_all.json").read_text())

object_ids = sorted({record["object_id"] for record in records})

assert len(object_ids) == 100

rng = random.Random(SEED)
rng.shuffle(object_ids)

splits = {
    "train": set(object_ids[:70]),
    "val": set(object_ids[70:85]),
    "test": set(object_ids[85:])
}

assert len(splits["train"]) == 70
assert len(splits["val"]) == 15
assert len(splits["test"]) == 15

assert not (splits["train"] & splits["val"])
assert not (splits["train"] & splits["test"])
assert not (splits["val"] & splits["test"])

for name, ids in splits.items():
    subset = [
        record for record in records
        if record["object_id"] in ids
    ]

    counts = Counter(record["object_id"] for record in subset)

    assert len(counts) == len(ids)
    assert all(count == 8 for count in counts.values())

    output = root / f"shapenet100_{name}.json"
    assert not output.exists(), f"File already exists: {output}"

    output.write_text(json.dumps(subset, indent=2))

    print(
        name.upper(),
        "OBJECTS:", len(ids),
        "IMAGES:", len(subset),
        "FILE:", output
    )

print("SHAPENET100_OBJECT_DISJOINT_SPLITS_COMPLETE")

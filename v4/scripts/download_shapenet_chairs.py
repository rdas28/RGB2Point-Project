from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import shutil

REPO_ID = "ShapeNet/shapenetcore-glb"
TARGET_COUNT = 100

output_dir = Path("../datasets/shapenetcore_glb/chair")
output_dir.mkdir(parents=True, exist_ok=True)

api = HfApi()

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

chair_files = sorted(
    path for path in files
    if path.startswith("chair/") and path.endswith(".glb")
)

existing = {
    path.name for path in output_dir.glob("*.glb")
}

print("AVAILABLE_CHAIRS:", len(chair_files))
print("EXISTING_CHAIRS:", len(existing))

needed = max(0, TARGET_COUNT - len(existing))
downloaded = 0
failed = 0

for path in chair_files:
    if downloaded >= needed:
        break

    filename = Path(path).name

    if filename in existing:
        continue

    try:
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=path,
            repo_type="dataset"
        )

        shutil.copy2(
            cached_path,
            output_dir / filename
        )

        downloaded += 1
        print(
            f"DOWNLOADED {downloaded}/{needed}: {filename}",
            flush=True
        )

    except Exception as exc:
        failed += 1
        print(
            f"FAILED: {filename}: {exc}",
            flush=True
        )

total = len(list(output_dir.glob("*.glb")))

print("\nDOWNLOAD_COMPLETE")
print("NEW_DOWNLOADS:", downloaded)
print("FAILED_ATTEMPTS:", failed)
print("TOTAL_LOCAL_CHAIRS:", total)

# RGB2Point v4 — Procedural Dataset Generation

## Overview

RGB2Point v4 uses a synthetic dataset generated with Blender for single-view 3D point cloud reconstruction.

The dataset contains:

- **20 objects:** 10 chairs and 10 tables
- **160 RGB images:** 8 camera views per object
- **Image resolution:** 224 × 224 pixels
- **Ground-truth point clouds:** 1,024 XYZ points per object
- **Coordinate system:** Z-up
- **Splitting strategy:** Object-disjoint training, validation, and testing

## 1. Environment

The following environment was verified on macOS Apple Silicon:

- Blender 5.2.3 LTS
- NumPy 2.3.4 inside Blender
- Python 3.11 for the training environment
- Blender Python API (`bpy`)

Blender is invoked from the command line using `blender --background`.

All commands below should be executed from the **RGB2Point-Project repository root**.

## 2. Generate procedural meshes

```bash
blender --background --python v4/scripts/generate_procedural_meshes.py
```

This script generates 10 procedural chairs and 10 procedural tables using Blender geometry operations.

Expected output:

`../datasets/rgb2point_v4/meshes/procedural/`

A total of 20 `.obj` mesh files are generated.

## 3. Render multiview RGB images

```bash
blender --background --python v4/scripts/render_procedural_multiview.py
```

This script loads the generated meshes and renders eight RGB views per object.

Expected output:

`../datasets/rgb2point_v4/images/procedural/`

A total of 160 PNG images are generated at 224 × 224 resolution.

## 4. Sample ground-truth point clouds

```bash
blender --background --python v4/scripts/sample_procedural_points.py
```

This script samples 1,024 points from each mesh surface using triangle-area-weighted sampling.

Expected output:

`../datasets/rgb2point_v4/points/procedural/`

A total of 20 `.npy` files are generated, each containing an array of shape `(1024, 3)`.

## 5. Create object-disjoint dataset splits

```bash
python -m v4.scripts.create_procedural_splits
```

The split script uses a fixed random seed of 42 and creates the following partitions:

| Split | Unique objects | RGB images |
|---|---:|---:|
| Training | 14 | 112 |
| Validation | 4 | 32 |
| Test | 2 | 16 |
| **Total** | **20** | **160** |

The same object never appears in more than one split.

Metadata files are written to:

- `metadata/procedural_train.json`
- `metadata/procedural_val.json`
- `metadata/procedural_test.json`

All paths are relative to `../datasets/rgb2point_v4/`.

## 6. Dataset directory structure

```text
datasets/
└── rgb2point_v4/
    ├── meshes/
    │   └── procedural/
    │       ├── chair_000.obj
    │       └── ...
    ├── images/
    │   └── procedural/
    │       ├── chair_000_view_00.png
    │       └── ...
    ├── points/
    │   └── procedural/
    │       ├── chair_000.npy
    │       └── ...
    └── metadata/
        ├── procedural_train.json
        ├── procedural_val.json
        └── procedural_test.json
```

The dataset directory is located alongside the repository rather than inside it.

## 7. Training after dataset generation

Once the dataset and metadata are available, run:

```bash
python -m v4.scripts.train_multiobject
```

The training pipeline uses a frozen pretrained Vision Transformer encoder and a learnable-query Transformer decoder to predict 1,024 XYZ points.

The best checkpoint is saved locally to:

`v4/outputs/multiobject_best_model.pth`

## Reproducibility notes

- Random seeds are fixed in the procedural generation and splitting scripts.
- Generated datasets and large model checkpoints are not included in the GitHub repository.
- The commands may overwrite existing dataset files; back up the dataset before rerunning them.
- Blender 5.2.3 LTS and its Python dependencies were checked, but a complete fresh dataset-generation run has not yet been independently verified.
- The reported experiments use a small synthetic dataset; real-world generalization has not been established.

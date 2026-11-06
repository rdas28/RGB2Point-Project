import bpy
import json
import math
import hashlib
import traceback
from pathlib import Path

import numpy as np
from mathutils import Vector

ROOT = Path("../datasets/rgb2point_v8").resolve()
SOURCE = Path("../datasets/shapenetcore_glb_expansion").resolve()

IMAGE_DIR = ROOT / "images" / "shapenet"
POINT_DIR = ROOT / "points" / "shapenet"
META_DIR = ROOT / "metadata"

for folder in (IMAGE_DIR, POINT_DIR, META_DIR):
    folder.mkdir(parents=True, exist_ok=True)

records = []
failures = []

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

def get_meshes():
    return [o for o in bpy.context.scene.objects
            if o.type == "MESH"]

def normalize_meshes(meshes):
    # Apply each object's world transform before normalization.
    from mathutils import Matrix

    for obj in meshes:
        obj.data = obj.data.copy()
        obj.data.transform(obj.matrix_world)
        obj.matrix_world = Matrix.Identity(4)

    corners = [
        obj.matrix_world @ Vector(c)
        for obj in meshes
        for c in obj.bound_box
    ]

    low = np.min(np.array(corners), axis=0)
    high = np.max(np.array(corners), axis=0)

    center = (low + high) / 2
    extent = float(np.max(high - low))

    if extent <= 1e-8:
        raise ValueError("Degenerate object bounds")

    # Fit the largest dimension into [-0.5, 0.5].
    for obj in meshes:
        for vertex in obj.data.vertices:
            vertex.co = (vertex.co - Vector(center)) / extent

def sample_points(meshes, seed):
    vertices = []
    triangles = []

    for obj in meshes:
        mesh = obj.data
        mesh.calc_loop_triangles()
        offset = len(vertices)

        vertices.extend([
            tuple(obj.matrix_world @ v.co)
            for v in mesh.vertices
        ])

        triangles.extend([
            [offset + i for i in tri.vertices]
            for tri in mesh.loop_triangles
        ])

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)

    if len(triangles) == 0:
        raise ValueError("No mesh triangles")

    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]

    areas = 0.5 * np.linalg.norm(
        np.cross(b - a, c - a), axis=1
    )

    total = areas.sum()
    if total <= 0:
        raise ValueError("Zero surface area")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        len(triangles), 1024, p=areas / total
    )

    u = np.sqrt(rng.random(1024))
    v = rng.random(1024)

    points = (
        (1 - u[:, None]) * a[chosen]
        + (u * (1 - v))[:, None] * b[chosen]
        + (u * v)[:, None] * c[chosen]
    )

    return points.astype(np.float32)

def render_views(meshes, object_id):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 1.6
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(
        type="AREA", location=(2, -3, 4)
    )
    light = bpy.context.object
    light.data.energy = 450
    light.data.shape = "DISK"
    light.data.size = 4

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 224
    scene.render.resolution_y = 224
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    scene.render.film_transparent = False
    scene.world.color = (0.8, 0.8, 0.8)

    target = Vector((0, 0, 0))

    for i in range(8):
        angle = 2 * math.pi * i / 8

        camera.location = (
            2.7 * math.cos(angle),
            2.7 * math.sin(angle),
            0.95
        )

        direction = target - camera.location
        camera.rotation_euler = direction.to_track_quat(
            "-Z", "Y"
        ).to_euler()

        filename = f"{object_id}_view_{i:02d}.png"
        scene.render.filepath = str(IMAGE_DIR / filename)

        bpy.ops.render.render(write_still=True)

        records.append({
            "image": f"images/shapenet/{filename}",
            "points": f"points/shapenet/{object_id}.npy",
            "object_id": object_id,
            "category": "chair",
            "view_id": i
        })

selection = json.loads(
    Path("datasets/rgb2point_expansion/selected_chairs.json").read_text()
)

files = [
    SOURCE / filename
    for filename in selection["selected_files"]
]

files = files[:450]
print("SHAPENET_MODELS_FOUND:", len(files), flush=True)

for index, path in enumerate(files):
    object_id = f"chair_{path.stem}"

    point_path = POINT_DIR / f"{object_id}.npy"
    image_paths = [
        IMAGE_DIR / f"{object_id}_view_{i:02d}.png"
        for i in range(8)
    ]

    complete = point_path.exists() and all(
        p.exists() and p.stat().st_size > 0
        for p in image_paths
    )

    if complete:
        try:
            existing_points = np.load(point_path)
            complete = (
                existing_points.shape == (1024, 3)
                and np.isfinite(existing_points).all()
            )
        except Exception:
            complete = False

    if complete:
        for i in range(8):
            records.append({
                "image": f"images/shapenet/{object_id}_view_{i:02d}.png",
                "points": f"points/shapenet/{object_id}.npy",
                "object_id": object_id,
                "category": "chair",
                "view_id": i
            })
        print("SKIPPED_COMPLETE:", object_id, flush=True)
        continue

    try:
        clear_scene()
        bpy.ops.import_scene.gltf(filepath=str(path))
        meshes = get_meshes()

        if not meshes:
            raise ValueError("No meshes imported")

        normalize_meshes(meshes)

        seed = int(
            hashlib.sha256(
                object_id.encode()
            ).hexdigest()[:8], 16
        )

        points = sample_points(meshes, seed)

        if points.shape != (1024, 3):
            raise ValueError("Invalid point cloud shape")

        # Save only after geometry validation.
        np.save(POINT_DIR / f"{object_id}.npy", points)

        render_views(meshes, object_id)

        print(
            "PROCESSED:", object_id,
            "VIEWS: 8", flush=True
        )

    except Exception as exc:
        records = [
            r for r in records
            if r["object_id"] != object_id
        ]
        failures.append({
            "object_id": object_id,
            "error": repr(exc)
        })
        print("FAILED:", object_id, repr(exc), flush=True)
        traceback.print_exc()

with (META_DIR / "shapenet_expansion_all.json").open("w") as f:
    json.dump(records, f, indent=2)

with (META_DIR / "shapenet_expansion_failures.json").open("w") as f:
    json.dump(failures, f, indent=2)

print("TOTAL_RECORDS:", len(records), flush=True)
print("TOTAL_FAILURES:", len(failures), flush=True)
print("SHAPENET_PREPROCESSING_COMPLETE", flush=True)

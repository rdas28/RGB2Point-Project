import bpy
import math
from pathlib import Path
from mathutils import Vector

ROOT = Path("../datasets/rgb2point_v4").resolve()
MESH_DIR = ROOT / "meshes" / "procedural"
IMAGE_DIR = ROOT / "images" / "procedural"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

mesh_files = sorted(MESH_DIR.glob("*.obj"))

print("MESHES_FOUND:", len(mesh_files), flush=True)

for mesh_path in mesh_files:

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.wm.obj_import(filepath=str(mesh_path))

    material = bpy.data.materials.new(
        f"{mesh_path.stem}_blue"
    )
    material.diffuse_color = (0.15, 0.35, 0.75, 1)
    material.use_nodes = True

    material.node_tree.nodes[
        "Principled BSDF"
    ].inputs["Base Color"].default_value = (
        0.15, 0.35, 0.75, 1
    )

    for obj in bpy.context.selected_objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(material)

    bpy.ops.object.camera_add()
    camera = bpy.context.object

    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2.4

    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(
        type="AREA",
        location=(2, -3, 4)
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

    # Frame each object around its actual bounding-box center.
    mesh_objects = [
        obj for obj in scene.objects
        if obj.type == "MESH"
    ]

    world_corners = [
        obj.matrix_world @ Vector(corner)
        for obj in mesh_objects
        for corner in obj.bound_box
    ]

    z_min = min(v.z for v in world_corners)
    z_max = max(v.z for v in world_corners)

    target = Vector((0, 0, (z_min + z_max) / 2))

    for i in range(8):

        angle = 2 * math.pi * i / 8

        camera.location = (
            2.7 * math.cos(angle),
            2.7 * math.sin(angle),
            target.z + 0.95
        )

        direction = target - camera.location

        camera.rotation_euler = direction.to_track_quat(
            "-Z", "Y"
        ).to_euler()

        output_path = IMAGE_DIR / (
            f"{mesh_path.stem}_view_{i:02d}.png"
        )

        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

    print(
        "OBJECT_RENDERED:",
        mesh_path.stem,
        "VIEWS: 8",
        flush=True
    )

print("PROCEDURAL_RENDERING_COMPLETE", flush=True)

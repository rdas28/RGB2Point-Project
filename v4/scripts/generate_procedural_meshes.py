import bpy
import os
import random

random.seed(42)

OUTPUT_DIR = os.path.abspath(
    "../datasets/rgb2point_v4/meshes/procedural"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def add_box(name, location, dimensions):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions

    bpy.ops.object.transform_apply(
        location=False,
        rotation=False,
        scale=True
    )

    return obj


def export_object(object_id):
    meshes = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH"
    ]

    bpy.ops.object.select_all(action="DESELECT")

    for obj in meshes:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()

    output = os.path.join(
        OUTPUT_DIR,
        f"{object_id}.obj"
    )

    bpy.ops.wm.obj_export(filepath=output)

    print("MESH_SAVED:", object_id)


def generate_chair(index):
    reset_scene()

    width = random.uniform(0.65, 0.95)
    depth = random.uniform(0.65, 0.90)
    seat_height = random.uniform(0.42, 0.58)
    back_height = random.uniform(0.60, 0.95)
    leg_size = random.uniform(0.07, 0.13)

    seat_thickness = 0.12

    add_box(
        "Seat",
        (0, 0, seat_height),
        (width, depth, seat_thickness)
    )

    add_box(
        "Backrest",
        (
            0,
            depth / 2 - 0.06,
            seat_height + back_height / 2
        ),
        (width, 0.12, back_height)
    )

    for x in [-1, 1]:
        for y in [-1, 1]:
            add_box(
                "Leg",
                (
                    x * (width / 2 - leg_size / 2),
                    y * (depth / 2 - leg_size / 2),
                    (seat_height - seat_thickness / 2) / 2
                ),
                (
                    leg_size,
                    leg_size,
                    seat_height - seat_thickness / 2
                )
            )

    export_object(f"chair_{index:03d}")


def generate_table(index):
    reset_scene()

    width = random.uniform(0.85, 1.30)
    depth = random.uniform(0.65, 1.05)
    height = random.uniform(0.65, 0.95)
    top_thickness = random.uniform(0.07, 0.12)
    leg_size = random.uniform(0.07, 0.13)

    add_box(
        "TableTop",
        (0, 0, height - top_thickness / 2),
        (width, depth, top_thickness)
    )

    leg_height = height - top_thickness

    for x in [-1, 1]:
        for y in [-1, 1]:
            add_box(
                "Leg",
                (
                    x * (width / 2 - leg_size / 2),
                    y * (depth / 2 - leg_size / 2),
                    leg_height / 2
                ),
                (leg_size, leg_size, leg_height)
            )

    export_object(f"table_{index:03d}")


for i in range(10):
    generate_chair(i)
    generate_table(i)

print("PROCEDURAL_GENERATION_COMPLETE")
print("TOTAL_OBJECTS:", 20)

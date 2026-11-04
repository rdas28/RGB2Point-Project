import bpy
import os

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

def add_box(name, location, scale):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(
        location=False, rotation=False, scale=True
    )
    return obj

# Seat
add_box("Seat", (0, 0, 0.5), (0.8, 0.8, 0.12))

# Backrest
add_box("Backrest", (0, 0.35, 0.9), (0.8, 0.12, 0.8))

# Four legs
for x in [-0.32, 0.32]:
    for y in [-0.32, 0.32]:
        add_box(
            "Leg",
            (x, y, 0.23),
            (0.10, 0.10, 0.46)
        )

# Join the chair parts into one mesh object
bpy.ops.object.select_all(action='SELECT')
bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
bpy.ops.object.join()

chair = bpy.context.object
chair.name = "TestChair"

output_path = os.path.abspath(
    "../datasets/rgb2point_v4/meshes/test_chair.obj"
)

bpy.ops.wm.obj_export(filepath=output_path)

print("CHAIR_MESH_CREATED:", output_path)
print("VERTEX_COUNT:", len(chair.data.vertices))
print("FACE_COUNT:", len(chair.data.polygons))

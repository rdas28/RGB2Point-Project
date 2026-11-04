import bpy
import math
import os
from mathutils import Vector

mesh_path = os.path.abspath(
    "../datasets/rgb2point_v4/meshes/test_chair.obj"
)
output_dir = os.path.abspath(
    "../datasets/rgb2point_v4/images"
)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.wm.obj_import(filepath=mesh_path)

material = bpy.data.materials.new("ChairBlue")
material.diffuse_color = (0.15, 0.35, 0.75, 1)
material.use_nodes = True
material.node_tree.nodes["Principled BSDF"].inputs[
    "Base Color"
].default_value = (0.15, 0.35, 0.75, 1)

for obj in bpy.context.selected_objects:
    if obj.type == 'MESH':
        obj.data.materials.clear()
        obj.data.materials.append(material)

bpy.ops.object.camera_add()
camera = bpy.context.object
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 1.8
bpy.context.scene.camera = camera

bpy.ops.object.light_add(type='AREA', location=(2, -3, 4))
light = bpy.context.object
light.data.energy = 450
light.data.shape = 'DISK'
light.data.size = 4

scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 224
scene.render.resolution_y = 224
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'

target = Vector((0, 0, 0.65))

for i in range(8):
    angle = 2 * math.pi * i / 8

    camera.location = (
        2.7 * math.cos(angle),
        2.7 * math.sin(angle),
        1.6
    )

    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat(
        '-Z', 'Y'
    ).to_euler()

    output_path = os.path.join(
        output_dir,
        f"test_chair_view_{i:02d}.png"
    )

    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    print(f"VIEW_{i:02d}_SAVED:", output_path)

print("MULTIVIEW_RENDERING_COMPLETE")

import bpy
import math
import os
from mathutils import Vector

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

mesh_path = os.path.abspath(
    "../datasets/rgb2point_v4/meshes/test_chair.obj"
)
output_path = os.path.abspath(
    "../datasets/rgb2point_v4/images/test_chair_view_00.png"
)

# Import the chair
bpy.ops.wm.obj_import(filepath=mesh_path)

# Material
material = bpy.data.materials.new("ChairBlue")
material.diffuse_color = (0.15, 0.35, 0.75, 1.0)
material.use_nodes = True
material.node_tree.nodes.get("Principled BSDF").inputs[
    "Base Color"
].default_value = (0.15, 0.35, 0.75, 1.0)

for obj in bpy.context.selected_objects:
    if obj.type == 'MESH':
        obj.data.materials.clear()
        obj.data.materials.append(material)

# Camera
bpy.ops.object.camera_add(location=(1.8, -2.4, 1.5))
camera = bpy.context.object
direction = Vector((0, 0, 0.55)) - camera.location
camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 1.8
bpy.context.scene.camera = camera

# Soft lighting
bpy.ops.object.light_add(
    type='AREA',
    location=(2, -3, 4)
)
light = bpy.context.object
light.data.energy = 450
light.data.shape = 'DISK'
light.data.size = 4

# World background
world = bpy.context.scene.world
world.color = (0.8, 0.8, 0.8)

# Render settings
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 224
scene.render.resolution_y = 224
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = False
scene.render.filepath = output_path

bpy.ops.render.render(write_still=True)

print("RENDER_SAVED:", output_path)

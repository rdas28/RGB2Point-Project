import bpy
import numpy as np
import os

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

mesh_path = os.path.abspath(
    "../datasets/rgb2point_v4/meshes/test_chair.obj"
)
output_path = os.path.abspath(
    "../datasets/rgb2point_v4/points/test_chair.npy"
)

bpy.ops.wm.obj_import(filepath=mesh_path)

vertices = []
triangles = []

for obj in bpy.context.selected_objects:
    if obj.type != 'MESH':
        continue

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

a = vertices[triangles[:, 0]]
b = vertices[triangles[:, 1]]
c = vertices[triangles[:, 2]]

areas = 0.5 * np.linalg.norm(
    np.cross(b - a, c - a), axis=1
)

rng = np.random.default_rng(42)

chosen = rng.choice(
    len(triangles),
    size=1024,
    p=areas / areas.sum()
)

u = np.sqrt(rng.random(1024))
v = rng.random(1024)

points = (
    (1 - u[:, None]) * a[chosen]
    + (u * (1 - v))[:, None] * b[chosen]
    + (u * v)[:, None] * c[chosen]
)

points = points.astype(np.float32)

np.save(output_path, points)

print("POINT_CLOUD_SAVED:", output_path)
print("POINT_CLOUD_SHAPE:", points.shape)
print("MIN_XYZ:", points.min(axis=0))
print("MAX_XYZ:", points.max(axis=0))

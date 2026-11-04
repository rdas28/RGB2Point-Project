import numpy as np
import matplotlib.pyplot as plt

points = np.load(
    "../datasets/rgb2point_v4/points/test_chair.npy"
)

fig = plt.figure(figsize=(12, 5))

# 3D view
ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    s=3,
    c=points[:, 2],
    cmap="viridis"
)
ax1.set_title("Chair Ground Truth - 3D")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z (Up)")
ax1.set_box_aspect((0.8, 0.81, 1.3))
ax1.view_init(elev=20, azim=-60)

# Front projection: X-Z
ax2 = fig.add_subplot(122)
ax2.scatter(
    points[:, 0],
    points[:, 2],
    s=4,
    alpha=0.7
)
ax2.set_title("Chair Ground Truth - Front (XZ)")
ax2.set_xlabel("X")
ax2.set_ylabel("Z (Up)")
ax2.set_aspect("equal")
ax2.grid(alpha=0.2)

plt.tight_layout()

output = "v4/outputs/test_chair_pointcloud.png"
plt.savefig(output, dpi=180)
print("VISUALIZATION_SAVED:", output)
plt.show()

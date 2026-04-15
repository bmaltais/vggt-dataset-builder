from hole_filling_renderer import HoleFillingRenderer
import numpy as np

width, height = 256, 256
r = HoleFillingRenderer(width, height)
r.confidence_threshold = 1.0

# In points.vert:
# cam_pos = u_view_mat * vec4(a_position, 1.0);
# gl_Position = u_proj_mat * cam_pos;

# Let's try to put the point squarely in the view.
# Standard OpenGL: looking down -Z.
# Let's put point at (0,0, -5)
# View matrix: identity (camera at origin, looking towards -Z)
# Projection: Perspective or Ortho.
# Let's use an ortho-like projection that maps [-1,1] to NDC.

pts = np.array([[0.0, 0.0, -0.5]], dtype=np.float32)
cols = np.array([[1.0, 0.5, 0.25]], dtype=np.float32)
confs = np.array([2.0], dtype=np.float32)

view = np.eye(4, dtype=np.float32)
proj = np.eye(4, dtype=np.float32)

# Identity proj will keep (0,0,-0.5) as (0,0,-0.5) in NDC.
# It should be visible.

# Maybe s0 is too small?
r.s0 = 0.1

img = r.render(pts, cols, confs, view, proj, 1.0)

print(f"Shape: {img.shape}")
print(f"Max value in image: {img.max()}")
print(f"Mean value in image: {img.mean():.4f}")

# Find where the point is
coords = np.where(img.any(axis=-1))
if coords[0].size > 0:
    print(f"Found non-zero pixels at {len(coords[0])} locations")
    idx = 0
    y, x = coords[0][idx], coords[1][idx]
    print(f"Pixel at ({y}, {x}): {img[y, x]}")
    expected = (cols[0] * 255).astype(np.uint8)
    print(f"Expected color: {expected}")
else:
    print("No non-zero pixels found.")

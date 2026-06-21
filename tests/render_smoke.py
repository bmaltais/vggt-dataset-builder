from hole_filling_renderer import HoleFillingRenderer
import numpy as np

# Set confidence threshold low so the point isn't filtered
r = HoleFillingRenderer(64, 64, confidence_threshold=0.5)
pts = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
cols = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
confs = np.array([1.0], dtype=np.float32)
view = np.eye(4, dtype=np.float32)
proj = np.eye(4, dtype=np.float32)
img = r.render(pts, cols, confs, view, proj, 1.0)
print("ok", img.shape, img.dtype, img.mean(), img.max())

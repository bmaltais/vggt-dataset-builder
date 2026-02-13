from pathlib import Path
import importlib.util
import sys
import numpy as np

# Load build_warp_dataset module by path to avoid import issues
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_warp_dataset", str(root / "build_warp_dataset.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
import types
# Provide a minimal dummy for hole_filling_renderer to avoid importing heavy runtime deps
sys.modules["hole_filling_renderer"] = types.SimpleNamespace(HoleFillingRenderer=object)
spec.loader.exec_module(module)

save_frame_cache = module.save_frame_cache
load_frame_cache = module.load_frame_cache
_cache_file_for_image = module._cache_file_for_image

scene_cache = Path('.cache') / 'smoke_scene'
scene_cache.mkdir(parents=True, exist_ok=True)
cache_path = scene_cache / 'test_image.npz'

frame = {
    'points': np.random.rand(5, 3).astype(np.float32),
    'colors': np.random.rand(5, 3).astype(np.float32),
    'confidences': np.random.rand(5).astype(np.float32),
    's0': 0.123,
}

print('Saving cache to', cache_path)
save_frame_cache(cache_path, frame)
loaded = load_frame_cache(cache_path)
print('Loaded keys:', list(loaded.keys()))
print('points shape:', loaded['points'].shape)
print('colors shape:', loaded['colors'].shape)
print('confidences shape:', loaded['confidences'].shape)
print('s0:', loaded.get('s0'))

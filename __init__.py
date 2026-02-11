try:
    try:
        from .vggt_comfy_nodes import VGGT_Model_Inference
    except ImportError:
        from vggt_comfy_nodes import VGGT_Model_Inference
except (ImportError, ModuleNotFoundError, ValueError):
    # Module not available or being imported in testing context
    VGGT_Model_Inference = None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if VGGT_Model_Inference is not None:
    NODE_CLASS_MAPPINGS["VGGT_Model_Inference"] = VGGT_Model_Inference
    NODE_DISPLAY_NAME_MAPPINGS["VGGT_Model_Inference"] = "VGGT Model Inference"

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

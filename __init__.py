try:
    from .vggt_comfy_nodes import VGGT_Model_Inference
except ImportError:
    from vggt_comfy_nodes import VGGT_Model_Inference

NODE_CLASS_MAPPINGS = {
    "VGGT_Model_Inference": VGGT_Model_Inference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VGGT_Model_Inference": "VGGT Model Inference",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:
    from .vggt_comfy_nodes import VGGT_PLY_Loader, VGGT_PLY_Viewer, VGGT_PLY_Renderer
except ImportError:
    from vggt_comfy_nodes import VGGT_PLY_Loader, VGGT_PLY_Viewer, VGGT_PLY_Renderer

NODE_CLASS_MAPPINGS = {
    "VGGT_PLY_Loader": VGGT_PLY_Loader,
    "VGGT_PLY_Viewer": VGGT_PLY_Viewer,
    "VGGT_PLY_Renderer": VGGT_PLY_Renderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VGGT_PLY_Loader": "VGGT PLY Loader",
    "VGGT_PLY_Viewer": "VGGT PLY Viewer",
    "VGGT_PLY_Renderer": "VGGT PLY Renderer",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

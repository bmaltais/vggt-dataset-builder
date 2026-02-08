import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock folder_paths
sys.modules["folder_paths"] = MagicMock()
sys.modules["folder_paths"].get_output_directory.return_value = "/tmp"

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    import vggt_comfy_nodes
    print("Successfully imported vggt_comfy_nodes")

    inference = vggt_comfy_nodes.VGGT_Model_Inference()
    print("Successfully initialized VGGT_Model_Inference")

    loader = vggt_comfy_nodes.VGGT_PLY_Loader()
    print("Successfully initialized VGGT_PLY_Loader")

    viewer = vggt_comfy_nodes.VGGT_PLY_Viewer()
    print("Successfully initialized VGGT_PLY_Viewer")

    renderer = vggt_comfy_nodes.VGGT_PLY_Renderer()
    print("Successfully initialized VGGT_PLY_Renderer")

    import __init__
    print("Successfully imported __init__")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

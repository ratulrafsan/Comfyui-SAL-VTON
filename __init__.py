import os
import folder_paths
from .vton_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

folder_paths.add_model_folder_path("salvton", os.path.join(folder_paths.models_dir, "salvton"))


try:
    model_path = folder_paths.get_folder_paths("salvton")[0]
    if not os.path.exists(model_path):
        os.makedirs(model_path)
except:
    print('Failed to create SALVTON model folder.')


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

import os
import random

import numpy as np
import torch

from .sal import inferSAL
from PIL import Image, ImageOps, ImageSequence
import folder_paths


node_category = "Clothing - SAL-VTON"

class SALVTONApply:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "garment": ("IMAGE", ),
                "person": ("IMAGE",),
                "garment_mask": ("IMAGE",)
            }
        }

    CATEGORY = node_category

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "apply_salvaton"

    def apply_salvaton(self, garment, person, garment_mask):
        return (inferSAL(folder_paths.get_folder_paths('salvton')[0], person, garment, garment_mask),)


class RandomImageFromDir:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "multiline": False,
                    "default": "./input"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = node_category

    def load_image(self, folder_path):
        files = os.listdir(folder_path)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".ico", ".jfif"}
        images = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]

        random_image = os.path.join(folder_path, random.choice(images))
        img = Image.open(random_image)
        output_images = []
        output_masks = []
        # this is from load_image node. Lazy but works :')
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "SALVTON_Apply": SALVTONApply,
    "SV_random": RandomImageFromDir,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SALVTON_Apply": "Apply SAL-VTON",
    "SV_random": "Random Image From Directory"
}
"""Segmentation module design 

segementor = Sam2Segmentor()

# segment an image
masks = segmentor.segment_image(rgb)

# or with prompts 
mask = segmentor.segment_with_point(rgb, point = (x,y))


# 1. Initialize (loads model weights)
  from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

  sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
  mask_generator = SamAutomaticMaskGenerator(sam)

  # 2. Generate masks
  masks = mask_generator.generate(image)  # image must be (H, W, 3) in [0, 255]

  # 3. Each mask is a dict with:
  # {
  #   'segmentation': (H, W) bool array,  # The actual mask
  #   'area': int,                        # Number of pixels
  #   'bbox': [x, y, w, h],              # Bounding box
  #   'predicted_iou': float,            # Quality score
  #   'stability_score': float,          # Confidence
  # }

SAM-2 Segmentation Module

Provides interface for segmenting objects in RGB images using Meta's SAM-2 model.
"""
import os
import numpy as np 
from typing import List, Dict, Tuple, Optional 
from pathlib import Path 
from urllib.request import urlretrieve


models_dict = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth"
}

class Sam2Segmentor:
    def __init__(
            self, 
            model_type: str = "vit_h", # vit_h is the highest quality, vit_l is medium, vit_b is low 
            device: str = "cpu",
    ):
        # path to weights 
        models_dir = Path(__file__).parent.parent.parent / "models"
        models_dir.mkdir(exist_ok = True)

        model_path = models_dict[model_type]
        checkpoint_path = models_dir / model_path

        # if weights don't exist, download the weights 
        if not checkpoint_path.exists():
            print(f"downloading {model_type} weights")
            self._download_model_weights(model_type, checkpoint_path)

        self.model_type = model_type 
        self.device = device

    def _download_model_weights(self,
            model_type: str, 
            checkpoint_path: str,
    ):
        base_url = "https://dl.fbaipublicfiles.com/segment_anything/"
        model_path = models_dict[model_type]
        full_path = base_url + model_path

        urlretrieve(full_path, checkpoint_path)







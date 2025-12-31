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
# import os
import numpy as np 
from typing import List, Dict, Tuple, Union, Optional 
from pathlib import Path 
from urllib.request import urlretrieve
from tqdm import tqdm

import sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# this is only for SAM v1 
# models_dict = {
#     "vit_h": "sam_vit_h_4b8939.pth",
#     "vit_l": "sam_vit_l_0b3195.pth",
#     "vit_b": "sam_vit_b_01ec64.pth"
# }

# for Sam 2.0 (configs are in sam2 package root)
config_files = {
    "tiny": "sam2_hiera_t.yaml",
    "small": "sam2_hiera_s.yaml",
    "base_plus": "sam2_hiera_b+.yaml",
    "large": "sam2_hiera_l.yaml",
}

checkpoint_files = {
    "tiny": "sam2_hiera_tiny.pt",
    "small": "sam2_hiera_small.pt",
    "base_plus": "sam2_hiera_base_plus.pt",
    "large": "sam2_hiera_large.pt"
}

class Sam2Segmentor:
    def __init__(
            self, 
            model_type: str = "large", # vit_h is the highest quality, vit_l is medium, vit_b is low 
            device: str = "cpu",
    ):
        # path to weights 
        models_dir = Path(__file__).parent.parent.parent / "models"
        models_dir.mkdir(exist_ok = True)

        model_path = checkpoint_files[model_type]
        checkpoint_path = models_dir / model_path

        # if weights don't exist, download the weights 
        if not checkpoint_path.exists():
            print(f"downloading {model_type} weights")
            self._download_model_weights(model_type, checkpoint_path)

        self.model_type = model_type
        self.device = device

        # SAM-2 uses Hydra config - just pass the config filename (it searches in sam2 package)
        config_name = config_files[model_type].replace(".yaml", "")  # Remove .yaml extension

        # build the model
        self.model = build_sam2(config_name, str(checkpoint_path), device=self.device)

        # create the mask generator 
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)


    def _download_model_weights(self,
            model_type: str, 
            checkpoint_path: Union[str, Path],
    ):
        if model_type not in checkpoint_files:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from {list(checkpoint_files.keys())}")
        base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"  # SAM 2.0
        
        model_path = checkpoint_files[model_type]
        full_path = base_url + model_path

        # can print the status here using a callback function 
        pbar = None 
        last = 0 

        def report_hook(blocknum: int, blocksize: int, total_size: int):
            nonlocal pbar, last

            downloaded = blocksize * blocknum
            if pbar is None:
                total = total_size if total_size and total_size > 0 else None
                pbar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {model_type}",
                )
            delta = downloaded - last

            if delta > 0:
                pbar.update(delta)
                last = downloaded

        try:
            urlretrieve(full_path, str(checkpoint_path), reporthook=report_hook)
        
        # this exists because resources must be released (files, sockets, user memory)
        finally:
            if pbar is not None:
                pbar.close()

    @staticmethod # signals that this method is not dependant on the object state 
    def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {rgb.shape}")
        
        # uint8 is a data type which takes 8 bits of memory representing numbers from 0 - 255 
        # 2^8 - 1 = 255 
        if rgb.dtype == np.uint8:
            img = rgb
        else:
            img = rgb.astype(np.float32)
            if img.max() <= 1.5:
                img *= 255.0 # scale if [0, 1] range 
            img = np.clip(img, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(img)

    def segment_image(self, rgb: np.ndarray) -> List[Dict]:
        # takes an RGB image, calls the self.mask_generator.generate()
        # The image must be in [0, 255] uint8 format, not [0, 1] float
        # return list of mask dictionaries 

        rgb_uint8 = self._to_uint8_rgb(rgb)
        list_masks = self.mask_generator.generate(rgb_uint8)
        return list_masks

    def segment_with_point(self, rgb: np.ndarray, point: np.ndarray):
        # complex method requiring (SAM2ImagePredictor API)
        # TO DO for later. 
        raise NotImplementedError("Use SAM2ImagePredictor for point/box prompts.")




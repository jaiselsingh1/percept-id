# import sys 
# sys.path.append("src")

from data_loaders.rgbd_loader import RGBDDataLoader
from segmentation.sam2_segmentor import Sam2Segmentor
import matplotlib.pyplot as plt 
import numpy as np 
from pathlib import Path

# Load RGBD image 
project_root = Path(__file__).parent.parent 
data_path = project_root / "data"

loader = RGBDDataLoader(data_root=str(data_path))
frame_data = loader.get_frame_data(0)

# Initialize the segmentor 
segmentor = Sam2Segmentor(model_type = "tiny")

# Run segmentation 
img = frame_data['cameras']['front']['rgb']
mask_list = segmentor.segment_image(img)

# visualize the results 
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")

print(f"detected {len(mask_list)} objects")

plt.subplot(1, 2, 2)
plt.imshow(img)

for i, mask_dict in enumerate(mask_list):
    # get the boolean mask 
    mask = mask_dict["segmentation"] # (H, W) boolean array 

    # generate random color 
    color = np.random.rand(3) # random RGB between [0, 1]
    
    colored_mask = np.zeros_like(img)
    colored_mask[mask] = color 

    plt.imshow(colored_mask, alpha=0.4)

plt.title(f"segmented : {len(mask_list)} objects")
plt.axis("off")

plt.tight_layout()
# plt.imshow() is putting an image into a figure 
# plt.show() is just show the figures
plt.show()

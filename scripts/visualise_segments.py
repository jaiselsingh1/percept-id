from data_loaders.rgbd_loader import RGBDDataLoader
from segmentation.sam2_segmentor import Sam2Segmentor
import matplotlib.pyplot as plt 
import numpy as np 
from pathlib import Path 

project_root = Path(__file__).parent.parent 
data_path = project_root / "data"

loader = RGBDDataLoader(data_root = str(data_path))
frame_idx = 0 

segmentor = Sam2Segmentor(model_type = "tiny")
frame_data = loader.get_frame_data(frame_idx, camera_name = "front")
cam_data = frame_data["cameras"]["front"]

rgb = cam_data["rgb"]
masks = segmentor.segment_image(rgb)
masks_sorted = sorted(masks, key = lambda m: m["area"], reverse = True)

print(f" found {len(masks_sorted)} segments")

# create a grid to visualise all the segments 
n_segments = min(6, len(masks_sorted)) # top 6 
fig, axes = plt.subplots(2, 3, figsize = (15, 10))
axes = axes.flatten()

for idx in range(n_segments):
    mask = masks_sorted[idx]["segmentation"]
    area = masks_sorted[idx]["area"]

    # create an overlay 
    overlay = rgb.copy()
    # this is alpha blending a pure green color and the RGB image
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

    axes[idx].imshow(overlay.astype(np.uint8))
    axes[idx].set_title(f"segment {idx} - area: {area}")
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig(project_root / "outputs" / "visualizations" / "segments_overview.png", dpi=150, bbox_inches='tight')
plt.show()


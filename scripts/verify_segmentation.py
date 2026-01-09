from data_loaders.rgbd_loader import RGBDDataLoader
from segmentation.sam2_segmentor import Sam2Segmentor
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
data_path = project_root / "data"

loader = RGBDDataLoader(data_root=str(data_path))
frame_idx = 0

segmentor = Sam2Segmentor(model_type="tiny")

frame_data = loader.get_frame_data(frame_idx, camera_name='front')
cam_data = frame_data['cameras']['front']

rgb = cam_data['rgb']
depth = cam_data['depth']

masks = segmentor.segment_image(rgb)
masks_sorted = sorted(masks, key=lambda m: m['area'], reverse=True)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original image
axes[0, 0].imshow(rgb)
axes[0, 0].set_title('Original RGB Image')
axes[0, 0].axis('off')

# Depth image
axes[0, 1].imshow(depth, cmap='viridis')
axes[0, 1].set_title('Depth Image')
axes[0, 1].axis('off')

# All segments overlay
all_masks_overlay = rgb.copy()
for idx, mask_info in enumerate(masks_sorted[:5]):
    mask = mask_info['segmentation']
    color = plt.cm.tab10(idx)[:3]
    all_masks_overlay[mask] = all_masks_overlay[mask] * 0.5 + np.array(color) * 255 * 0.5
axes[0, 2].imshow(all_masks_overlay.astype(np.uint8))
axes[0, 2].set_title('Top 5 Segments Overlay')
axes[0, 2].axis('off')

# Rack (segment 0)
rack_mask = masks_sorted[0]['segmentation']
rack_overlay = rgb.copy()
rack_overlay[rack_mask] = rack_overlay[rack_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
axes[1, 0].imshow(rack_overlay.astype(np.uint8))
axes[1, 0].set_title(f'Segment 0: RACK (area={masks_sorted[0]["area"]})')
axes[1, 0].axis('off')

# Mug (segment 1)
mug_mask = masks_sorted[1]['segmentation']
mug_overlay = rgb.copy()
mug_overlay[mug_mask] = mug_overlay[mug_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
axes[1, 1].imshow(mug_overlay.astype(np.uint8))
axes[1, 1].set_title(f'Segment 1: MUG (area={masks_sorted[1]["area"]})')
axes[1, 1].axis('off')

# Just the masks
axes[1, 2].imshow(rack_mask, cmap='gray', alpha=0.5)
axes[1, 2].imshow(mug_mask, cmap='Reds', alpha=0.5)
axes[1, 2].set_title('Rack (gray) + Mug (red) masks')
axes[1, 2].axis('off')

plt.tight_layout()
output_path = project_root / "outputs" / "visualizations" / "segmentation_verification.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization to: {output_path}")
plt.show()

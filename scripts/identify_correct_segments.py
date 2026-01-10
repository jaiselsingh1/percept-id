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
masks = segmentor.segment_image(rgb)
masks_sorted = sorted(masks, key=lambda m: m['area'], reverse=True)

# Show top 10 segments with labels
n_segments = min(10, len(masks_sorted))
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx in range(n_segments):
    mask = masks_sorted[idx]['segmentation']
    area = masks_sorted[idx]['area']

    # Create overlay
    overlay = rgb.copy()
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 1, 0]) * 0.5

    axes[idx].imshow(overlay.astype(np.float32))
    axes[idx].set_title(f"Segment {idx}\nArea: {area} px", fontsize=10, fontweight='bold')
    axes[idx].axis('off')

plt.suptitle("IDENTIFY: Which segments are MUG and RACK?", fontsize=16, fontweight='bold')
plt.tight_layout()
output_path = project_root / "outputs" / "visualizations" / "all_segments_labeled.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to: {output_path}")
print("\nPlease identify:")
print("- Which segment number is the MUG?")
print("- Which segment number is the RACK?")
plt.show()

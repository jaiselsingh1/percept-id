from data_loaders.rgbd_loader import RGBDDataLoader
from data_loaders.point_cloud import create_point_cloud_from_rgbd
from segmentation.sam2_segmentor import Sam2Segmentor
import open3d as o3d
import numpy as np
from pathlib import Path

# load RGB-D data
project_root = Path(__file__).parent.parent
data_path = project_root / "data"

loader = RGBDDataLoader(data_root=str(data_path))
frame_data = loader.get_frame_data(0, camera_name='front')  # Use front camera

# extract data
rgb = frame_data['cameras']['front']['rgb']
depth = frame_data['cameras']['front']['depth']
K = frame_data['cameras']['front']['intrinsics']
world_T_camera = frame_data['cameras']['front']['extrinsics']

# run segmentation
segmentor = Sam2Segmentor(model_type="tiny")
masks = segmentor.segment_image(rgb)

print(f"\nFound {len(masks)} objects")

# sort masks by area and show top 5
masks_sorted = sorted(masks, key=lambda m: m['area'], reverse=True)

print("\nTop 5 objects by area:")
for i in range(min(5, len(masks_sorted))):
    mask = masks_sorted[i]
    bbox = mask['bbox']  # [x, y, w, h]
    print(f"  [{i}] Area={mask['area']:6d} pixels, "
          f"BBox=[x:{bbox[0]:.0f}, y:{bbox[1]:.0f}, w:{bbox[2]:.0f}, h:{bbox[3]:.0f}], "
          f"IoU={mask['predicted_iou']:.3f}")

# select which object to visualize
# Change this index to visualize different objects!
# 0 = largest (table), 1 = 2nd largest, etc.
selected_idx = 4

selected_mask = masks_sorted[selected_idx]
print(f"\n→ Visualizing object {selected_idx}")
print(f"  Area: {selected_mask['area']} pixels")
print(f"  BBox: {selected_mask['bbox']}")
print(f"  Quality (IoU): {selected_mask['predicted_iou']:.3f}")

# get the boolean mask (H, W)
seg_mask = selected_mask['segmentation']

print(f"\nMask stats:")
print(f"  Mask shape: {seg_mask.shape}")
print(f"  True pixels: {seg_mask.sum()} / {seg_mask.size} ({100*seg_mask.sum()/seg_mask.size:.1f}%)")

# create segmented point cloud
# flatten the mask (H, W) -> (H*W)
seg_mask_flat = seg_mask.reshape(-1)

pcd = create_point_cloud_from_rgbd(
    rgb=rgb,
    depth=depth,
    K=K,
    world_T_camera=world_T_camera,
    segmentation_mask=seg_mask_flat
)

print(f"\nPoint cloud stats:")
print(f"  Total points: {len(pcd.points)}")
print(f"  Has colors: {len(pcd.colors) > 0}")

# visualize
o3d.visualization.draw_geometries([pcd])

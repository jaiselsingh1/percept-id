from data_loaders.rgbd_loader import RGBDDataLoader
from data_loaders.point_cloud import create_point_cloud_from_rgbd
from segmentation.sam2_segmentor import Sam2Segmentor
import open3d as o3d
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
data_path = project_root / "data"

loader = RGBDDataLoader(data_root=str(data_path))
frame_data = loader.get_frame_data(0, camera_name='front')  # Use front camera

# Extract data
rgb = frame_data['cameras']['front']['rgb']
depth = frame_data['cameras']['front']['depth']
K = frame_data['cameras']['front']['intrinsics']
world_T_camera = frame_data['cameras']['front']['extrinsics']

# run the segmentation
segmentor = Sam2Segmentor(model_type="tiny")
masks = segmentor.segment_image(rgb)

print(f"Found {len(masks)} objects")

# pick the largest mask (likely the most prominent object)
largest_mask = max(masks, key=lambda m: m['area'])
print(f"Largest object has {largest_mask['area']} pixels")
print(f"Bounding box: {largest_mask['bbox']}")
print(f"Quality (IoU): {largest_mask['predicted_iou']:.3f}")

# get the boolean mask (H, W)
seg_mask = largest_mask['segmentation']

# create segmented point cloud
# need to flatten the mask (H, W) -> (H*W)
seg_mask_flat = seg_mask.reshape(-1)

pcd = create_point_cloud_from_rgbd(
    rgb=rgb,
    depth=depth,
    K=K,
    world_T_camera=world_T_camera,
    segmentation_mask=seg_mask_flat  # Pass flattened mask
)

print(f"Point cloud has {len(pcd.points)} points")
o3d.visualization.draw_geometries([pcd])
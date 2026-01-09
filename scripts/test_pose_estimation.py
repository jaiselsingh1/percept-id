from data_loaders.rgbd_loader import RGBDDataLoader
from data_loaders.point_cloud import create_point_cloud_from_rgbd
from segmentation.sam2_segmentor import Sam2Segmentor
from pose_estimation.icp_estimator import ICPPoseEstimator
from utils.transforms import transform_to_pose
import open3d as o3d
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
K = cam_data['intrinsics']
world_T_camera = cam_data['extrinsics']

masks = segmentor.segment_image(rgb)
masks_sorted = sorted(masks, key=lambda m: m['area'], reverse=True)

print(f"\nFound {len(masks)} objects")
print("Top 5 objects:")
for i in range(min(5, len(masks_sorted))):
    mask = masks_sorted[i]
    print(f"  [{i}] Area={mask['area']:6d} pixels")

target_idx = 1
print(f"\n→ Selected object {target_idx} as target")

seg_mask = masks_sorted[target_idx]['segmentation']

observed_pcd = create_point_cloud_from_rgbd(
    rgb=rgb,
    depth=depth,
    K=K,
    world_T_camera=world_T_camera,
    segmentation_mask=seg_mask
)

print(f"\nObserved point cloud: {len(observed_pcd.points)} points")

# Load CAD model
print("\n loading the CAD model")
model_path = project_root / "outputs" / "meshes" / "mug_and_zip" / "039_mug" / "0" / "039_mug_0.stl"
model_mesh = o3d.io.read_triangle_mesh(str(model_path))
print(f"Loaded mesh with {len(model_mesh.vertices)} vertices, {len(model_mesh.triangles)} triangles")

# Scale model to real-world size (STL is ~1.93m, real mug is ~0.12m)
scale_factor = 0.12 / 1.93
model_mesh.scale(scale_factor, center=model_mesh.get_center())
print(f"Scaled model by {scale_factor:.4f}x")

model_pcd = model_mesh.sample_points_uniformly(number_of_points=5000)

# Run pose estimation
print("\n running icp pose estimation")
estimator = ICPPoseEstimator(voxel_size=0.005, icp_threshold=0.02)
world_T_model, fitness, rmse = estimator.estimate_pose(model_pcd, observed_pcd)

print(f"ICP Fitness: {fitness:.4f}")
print(f"ICP RMSE: {rmse:.6f} m")

position, quaternion = transform_to_pose(world_T_model)

print(f"\n estimated object pose")
print(f"Position (x, y, z): {position}")
print(f"Quaternion (x, y, z, w): {quaternion}")

# Visualize
print("\n visualisation")
print("Gray = Observed point cloud")
print("Red = Aligned CAD model")

observed_vis = estimator.preprocess_point_cloud(observed_pcd)
observed_vis.paint_uniform_color([0.5, 0.5, 0.5])

model_aligned = model_pcd.transform(world_T_model)
model_vis = estimator.preprocess_point_cloud(model_aligned)
model_vis.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries([observed_vis, model_vis])

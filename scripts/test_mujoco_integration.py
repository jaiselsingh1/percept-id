from data_loaders.rgbd_loader import RGBDDataLoader
from data_loaders.point_cloud import create_point_cloud_from_rgbd
from segmentation.sam2_segmentor import Sam2Segmentor
from pose_estimation.icp_estimator import ICPPoseEstimator
from mujoco_integration.scene_builder import MuJoCoSceneBuilder
from utils.transforms import transform_to_pose
import open3d as o3d
import numpy as np
from pathlib import Path
import shutil

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

target_idx = 1
seg_mask = masks_sorted[target_idx]['segmentation']

observed_pcd = create_point_cloud_from_rgbd(
    rgb=rgb,
    depth=depth,
    K=K,
    world_T_camera=world_T_camera,
    segmentation_mask=seg_mask
)

print(f"Observed point cloud: {len(observed_pcd.points)} points")

# Load CAD model
model_path = project_root / "outputs" / "meshes" / "mug_and_zip" / "039_mug" / "0" / "039_mug_0.stl"
model_mesh = o3d.io.read_triangle_mesh(str(model_path))

scale_factor = 0.12 / 1.93
model_mesh.scale(scale_factor, center=model_mesh.get_center())

model_pcd = model_mesh.sample_points_uniformly(number_of_points=5000)

# Run pose estimation
print("\n=== Running pose estimation ===")
estimator = ICPPoseEstimator(voxel_size=0.005, icp_threshold=0.02)
world_T_model, fitness, rmse = estimator.estimate_pose(model_pcd, observed_pcd)

position, quaternion = transform_to_pose(world_T_model)

print(f"ICP Fitness: {fitness:.4f}")
print(f"ICP RMSE: {rmse:.6f} m")
print(f"Position (x, y, z): {position}")
print(f"Quaternion (x, y, z, w): {quaternion}")

# Build MuJoCo scene
scene_builder = MuJoCoSceneBuilder(output_path=str(project_root / "outputs" / "mujoco_scene.xml"))

scene_builder.add_object(
    name="mug",
    mesh_path=model_path,
    position=position,
    quaternion=quaternion,
    scale=scale_factor
)

scene_builder.save_scene()

# Copy mesh to meshes directory
meshdir = scene_builder.get_meshdir()
meshdir.mkdir(parents=True, exist_ok=True)
dest_mesh = meshdir / model_path.name
shutil.copy(model_path, dest_mesh)
print(f"Copied mesh to: {dest_mesh}")

print(f"Scene XML: {scene_builder.output_path}")
print(f"Load in MuJoCo viewer with: python -m mujoco.viewer {scene_builder.output_path}")

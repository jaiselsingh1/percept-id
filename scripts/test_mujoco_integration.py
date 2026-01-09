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

print(f"Found {len(masks_sorted)} segments")

# Define objects to process: (segment_idx, name, model_path, approximate_scale)
objects_to_process = [
    {
        'segment_idx': 1,
        'name': 'mug',
        'model_path': project_root / "outputs" / "meshes" / "mug_and_zip" / "039_mug" / "0" / "039_mug_0.stl",
        'scale': 0.36
    },
    {
        'segment_idx': 0,
        'name': 'rack',
        'model_path': project_root / "outputs" / "meshes" / "mug_and_zip" / "040_rack" / "0" / "040_rack_0.stl",
        'scale': 4.77
    }
]

# Initialize MuJoCo scene builder and pose estimator
scene_builder = MuJoCoSceneBuilder(output_path=str(project_root / "outputs" / "mujoco_scene.xml"))
estimator = ICPPoseEstimator(voxel_size=0.005, icp_threshold=0.02)

# Process each object
for obj_info in objects_to_process:
    segment_idx = obj_info['segment_idx']
    name = obj_info['name']
    model_path = obj_info['model_path']
    scale_factor = obj_info['scale']

    print(f"\n--- Processing {name} (segment {segment_idx}) ---")

    # Get segmentation mask
    seg_mask = masks_sorted[segment_idx]['segmentation']

    # Create observed point cloud from RGBD
    observed_pcd = create_point_cloud_from_rgbd(
        rgb=rgb,
        depth=depth,
        K=K,
        world_T_camera=world_T_camera,
        segmentation_mask=seg_mask
    )

    print(f"Observed point cloud: {len(observed_pcd.points)} points")

    # Load and scale CAD model
    model_mesh = o3d.io.read_triangle_mesh(str(model_path))
    model_mesh.scale(scale_factor, center=model_mesh.get_center())
    model_pcd = model_mesh.sample_points_uniformly(number_of_points=5000)

    # Run pose estimation
    world_T_model, fitness, rmse = estimator.estimate_pose(model_pcd, observed_pcd)
    position, quaternion = transform_to_pose(world_T_model)

    print(f"ICP Fitness: {fitness:.4f}")
    print(f"ICP RMSE: {rmse:.6f} m")
    print(f"Position (x, y, z): {position}")
    print(f"Quaternion (x, y, z, w): {quaternion}")

    # Add object to scene
    scene_builder.add_object(
        name=name,
        mesh_path=model_path,
        position=position,
        quaternion=quaternion,
        scale=scale_factor
    )

    # Copy mesh to meshes directory
    meshdir = scene_builder.get_meshdir()
    meshdir.mkdir(parents=True, exist_ok=True)
    dest_mesh = meshdir / model_path.name
    shutil.copy(model_path, dest_mesh)
    print(f"Copied mesh to: {dest_mesh}")

# Save the scene
scene_builder.save_scene()

print(f"\n=== Multi-object scene complete ===")
print(f"Scene XML: {scene_builder.output_path}")
print(f"Load in MuJoCo viewer with: python -m mujoco.viewer {scene_builder.output_path}")

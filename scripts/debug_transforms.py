from data_loaders.rgbd_loader import RGBDDataLoader
from data_loaders.point_cloud import create_point_cloud_from_rgbd
import open3d as o3d
import numpy as np
from pathlib import Path

def create_coordinate_frame(T, scale=0.1):
    """Create a coordinate frame mesh at transform T."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    frame.transform(T)
    return frame

project_root = Path(__file__).parent.parent
data_path = project_root / "data"

loader = RGBDDataLoader(data_root=str(data_path))
frame_idx = 0

print("\n front camera")
frame_data_front = loader.get_frame_data(frame_idx, camera_name='front')
world_T_camera_front = frame_data_front['cameras']['front']['extrinsics']

print("world_T_camera (extrinsics):")
print(world_T_camera_front)
print("\nCamera position in world:")
camera_T_world_front = np.linalg.inv(world_T_camera_front)
print(camera_T_world_front[:3, 3])

print("\n=== WRIST CAMERA ===")
frame_data_wrist = loader.get_frame_data(frame_idx, camera_name='wrist')
world_T_camera_wrist = frame_data_wrist['cameras']['wrist']['extrinsics']

print("world_T_camera (extrinsics):")
print(world_T_camera_wrist)
print("\nCamera position in world:")
camera_T_world_wrist = np.linalg.inv(world_T_camera_wrist)
print(camera_T_world_wrist[:3, 3])

print("\n robot state")
robot_state = loader.get_robot_state(frame_idx)
ee_pos = robot_state['obs.ee_pos']
ee_quat = robot_state['obs.ee_quat']
print(f"EE position: {ee_pos}")
print(f"EE quaternion: {ee_quat}")

# Create point clouds WITHOUT segmentation to see everything
print("\n creating point clouds")
rgb_front = frame_data_front['cameras']['front']['rgb']
depth_front = frame_data_front['cameras']['front']['depth']
K_front = frame_data_front['cameras']['front']['intrinsics']

rgb_wrist = frame_data_wrist['cameras']['wrist']['rgb']
depth_wrist = frame_data_wrist['cameras']['wrist']['depth']
K_wrist = frame_data_wrist['cameras']['wrist']['intrinsics']

pcd_front = create_point_cloud_from_rgbd(
    rgb=rgb_front,
    depth=depth_front,
    K=K_front,
    world_T_camera=world_T_camera_front,
    filter_invalid=True
)

pcd_wrist = create_point_cloud_from_rgbd(
    rgb=rgb_wrist,
    depth=depth_wrist,
    K=K_wrist,
    world_T_camera=world_T_camera_wrist,
    filter_invalid=True
)

# Color code the point clouds
pcd_front.paint_uniform_color([1, 0, 0])  # red = front camera
pcd_wrist.paint_uniform_color([0, 0, 1])  # blue = wrist camera

print(f"Front camera: {len(pcd_front.points)} points")
print(f"Wrist camera: {len(pcd_wrist.points)} points")

# Create coordinate frames for visualization
world_frame = create_coordinate_frame(np.eye(4), scale=0.2)  # world origin
front_cam_frame = create_coordinate_frame(camera_T_world_front, scale=0.1)
wrist_cam_frame = create_coordinate_frame(camera_T_world_wrist, scale=0.1)

print("\n=== Visualization ===")
print("Red = Front camera points")
print("Blue = Wrist camera points")
print("RGB axes = World frame (large)")
print("Small RGB axes = Camera frames")

o3d.visualization.draw_geometries([
    pcd_front,
    pcd_wrist,
    world_frame,
    front_cam_frame,
    wrist_cam_frame
])

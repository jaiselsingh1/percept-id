import sys 
sys.path.append('src')

from data_loaders.rgbd_loader import RGBDDataLoader
from data_loaders.point_cloud import create_point_cloud_from_rgbd
import numpy as np 
import open3d as o3d 
from pathlib import Path 

project_root = Path(__file__).parent.parent
data_path = project_root / "data"

loader = RGBDDataLoader(data_root=str(data_path))
frame_data = loader.get_frame_data(0)

# Define bounding box (table region from HW3)
bbox = np.array([
    [-0.5, -0.5, -1.0],  # min [x, y, z]
    [ 0.8,  0.5,  0.2]   # max [x, y, z]
])
pcds = []
for cam_name in ['front', 'wrist']:
    cam_data = frame_data['cameras'][cam_name]
    pcd = create_point_cloud_from_rgbd(
        rgb=cam_data['rgb'],
        depth=cam_data['depth'],
        K=cam_data['intrinsics'],
        world_T_camera=cam_data['extrinsics'],
        filter_invalid=True,
        bbox=bbox
    )
    pcds.append(pcd)
    print(f"{cam_name} camera: {len(pcd.points)} points")
merged = pcds[0] + pcds[1]
print(f"Merged: {len(merged.points)} total points")

o3d.visualization.draw_geometries([merged])



from data_loaders.rgbd_loader import RGBDDataLoader
from data_loaders.point_cloud import create_point_cloud_from_rgbd
from segmentation.sam2_segmentor import Sam2Segmentor
from utils.transforms import transform_to_pose
import open3d as o3d 
import numpy as np 
from pathlib import Path

def estimate_pose_with_icp(
        observed_pcd: o3d.geometry.PointCloud, 
        model_pcd: o3d.geometry.PointCloud, 
        initial_transform: np.ndarray = np.eye(4)
) -> np.ndarray:
    """this is a function to try and help estimate the object pose using ICP allignment"""

    threshold = 0.02 # 2 cm 

    result = o3d.pipelines.registration.registration_icp(
        source=model_pcd, 
        target=observed_pcd,
        max_correspondance_distance = threshold, 
        init = initial_transform,
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iterations = 2000)
    )

    print(f"ICP Result")
    print(f"Fitness: {result.fitness:.4f}")
    print(f"RMSE: {result.inlier_rmse:.6f} m")
    print(f" Converged: {result.fitness > 0.5}")

    return result.transformation

project_root = Path(__file__).parent.parent
data_path = project_root / "data"
loader = RGBDDataLoader(data_root = str(data_path))
frame_idx = 0 

segmentor = Sam2Segmentor(model_type = "tiny")
point_clouds = []

for cam_name in ["front", "wrist"]:
    print(f"Processing {cam_name}")

    frame_data = loader.get_frame_data(frame_idx, camera_name = cam_name)
    

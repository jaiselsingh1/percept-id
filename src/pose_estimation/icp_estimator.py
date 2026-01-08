import numpy as np 
import open3d as o3d 
from typing import Tuple, Optional 

class ICPPoseEstimator:
        """Estimate the 6DOF pose by aligning the CAD model to the observed point cloud"""
        def __init__(
            self, 
            voxel_size: float = 0.005,
            icp_threshold: float = 0.02, 
            max_iterations: int = 2000):
                self.voxel_size = voxel_size 
                self.icp_threshold = icp_threshold 
                self.max_iterations = max_iterations 

        def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
                # downsample and then estimate the normals 
                # voxels are basically points in a 3D space so the downsampling basically takes the N points and breaks them down 
        
                pcd_down = pcd.voxel_down_sample(self.voxel_size)
                pcd_down.estimate_normals(
                        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = self.voxel_size * 2, max_nn = 30)
                )
                return pcd_down
        
        def compute_initial_alignment(
                        self, 
                        model_pcd: o3d.geometry.PointCloud, 
                        observed_pcd: o3d.geometry.PointCloud
        ) -> np.ndarray:
                observed_center = observed_pcd.get_center() 
                model_center = model_pcd.get_center()

                initial_transform = np.eye(4)
                initial_transform[:3, 3] = observed_center - model_center 

                return initial_transform 
        
        def estimate_pose(
                        self, 
                        model_pcd: o3d.geometry.PointCloud, 
                        observed_pcd: o3d.geometry.PointCloud,
                        initial_transform: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, float, float]:
                # estimate the object pose using ICP 

                model_down = self.preprocess_point_cloud(model_pcd)
                observed_down = self.preprocess_point_cloud(observed_pcd)

                if initial_transform is None:
                        initial_transform = self.compute_initial_alignment(model_down, observed_down)

                result = o3d.pipelines.registration.registration_icp(
                        source = model_down,
                        target = observed_down,
                        max_correspondence_distance = self.icp_threshold,
                        init = initial_transform,
                        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = self.max_iterations)
                )
                return result.transformation, result.fitness, result.inlier_rmse

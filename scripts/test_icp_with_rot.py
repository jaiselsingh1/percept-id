from data_loaders.rgbd_loader import RGBDDataLoader
from data_loaders.point_cloud import create_point_cloud_from_rgbd
from segmentation.sam2_segmentor import Sam2Segmentor
from pose_estimation.icp_estimator import ICPPoseEstimator
import open3d as o3d 
import numpy as np 
from pathlib import Path 
from scipy.spatial.transform import Rotation 

project_root = Path(__file__).parent.parent
data_path = project_root / "data"
loader = RGBDDataLoader(data_root = str(data_path))
frame_idx = 0 
segmentor = Sam2Segmentor(model_type = "tiny")
frame_data = loader.get_frame_data(frame_idx, camera_name = 'front')
cam_data = frame_data["cameras"]["front"]
depth = cam_data["depth"]
rgb = cam_data["rgb"]
K = cam_data["intrinsics"]
world_T_camera = cam_data["extrinsics"]

# segment the image 
masks = segmentor.segment_image(rgb)
masks_sorted = sorted(masks, key = lambda m: m["area"], reverse = True)
# process the mug 
mug_mask = masks_sorted[1]["segmentation"]
mug_pcd_observed = create_point_cloud_from_rgbd(
    rgb = rgb, 
    depth = depth, 
    K = K, 
    world_T_camera = world_T_camera, 
    segmentation_mask = mug_mask
)

# load the mug CAD model 
mug_model_path = project_root / "outputs" / "meshes" / "mug_and_zip" / "039_mug" / "0" / "039_mug_0.stl"
mug_mesh = o3d.io.read_triangle_mesh(str(mug_model_path))

# try diff scales and orientations 
scales_to_try = [0.062, 0.1, 0.15, 0.2, 0.36, 0.5]
rot_to_try = [
    ("no rotation", [0, 0, 0]), 
    ("90° X", [90, 0, 0]),
    ("90° Y", [0, 90, 0]), 
    ("90° Z", [0, 0, 90]),
    ("-90° -X", [-90, 0, 0]),
    ("-90° -Y", [0, -90, 0]),
]

estimator = ICPPoseEstimator(voxel_size = 0.005, icp_threshold = 0.02)
best_fitness = 0
best_config = None

# loop to test
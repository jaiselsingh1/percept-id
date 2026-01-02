from data_loaders.rgbd_loader import RGBDDataLoader
from data_loaders.point_cloud import create_point_cloud_from_rgbd
from segmentation.sam2_segmentor import Sam2Segmentor
import open3d as o3d
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
data_path = project_root / "data"

loader = RGBDDataLoader(data_root=str(data_path))
frame_idx = 0
frame_data = loader.get_frame_data(frame_idx)

print(f"Loaded frame {frame_idx} with cameras: {list(frame_data['cameras'].keys())}")

segmentor = Sam2Segmentor(model_type="tiny")
point_clouds = []

for cam_name in ['front', 'wrist']:
    print(f"Processing {cam_name.upper()} camera")

    cam_data = frame_data['cameras'][cam_name]
    rgb = cam_data['rgb']
    depth = cam_data['depth']
    K = cam_data['intrinsics']
    camera_T_world = cam_data['extrinsics']

    masks = segmentor.segment_image(rgb)
    print(f"Found {len(masks)} objects")

    masks_sorted = sorted(masks, key=lambda m: m['area'], reverse=True)
    print(f"\nTop 5 objects:")
    for i in range(min(5, len(masks_sorted))):
        mask = masks_sorted[i]
        print(f"  [{i}] Area={mask['area']:6d}, BBox={mask['bbox']}")

    selected_idx = 4
    selected_mask = masks_sorted[selected_idx]

    print(f"\nSelected object {selected_idx}:")
    print(f"  Area: {selected_mask['area']} pixels")

    seg_mask = selected_mask['segmentation']
    seg_mask_flat = seg_mask.reshape(-1)

    pcd = create_point_cloud_from_rgbd(
        rgb=rgb,
        depth=depth,
        K=K,
        camera_T_world=camera_T_world,
        segmentation_mask=seg_mask_flat
    )

    print(f"Point cloud: {len(pcd.points)} points")

    if cam_name == 'wrist':
        colors = np.asarray(pcd.colors)
        colors[:, 2] = np.clip(colors[:, 2] * 1.2, 0, 1)
        # this is to go from np.ndarray to the C++ equivalent 
        # handles 64 bit floats (doubles)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    point_clouds.append(pcd)

merged_pcd = o3d.geometry.PointCloud()
for pcd in point_clouds:
    merged_pcd += pcd

print(f"Merged point cloud: {len(merged_pcd.points)} points")
print(f"  Front camera contribution: {len(point_clouds[0].points)} points")
print(f"  Wrist camera contribution: {len(point_clouds[1].points)} points")

o3d.visualization.draw_geometries([merged_pcd])

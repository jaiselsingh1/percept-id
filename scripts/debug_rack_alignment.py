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

segmentor = Sam2Segmentor(model_type="tiny")

frame_data = loader.get_frame_data(frame_idx, camera_name='front')
cam_data = frame_data['cameras']['front']

rgb = cam_data['rgb']
depth = cam_data['depth']
K = cam_data['intrinsics']
world_T_camera = cam_data['extrinsics']

# Segment the image
masks = segmentor.segment_image(rgb)
masks_sorted = sorted(masks, key=lambda m: m['area'], reverse=True)

# Get rack segment (segment 0)
rack_mask = masks_sorted[0]['segmentation']

# Create observed point cloud for rack
rack_pcd = create_point_cloud_from_rgbd(
    rgb=rgb,
    depth=depth,
    K=K,
    world_T_camera=world_T_camera,
    segmentation_mask=rack_mask
)

print(f"Rack observed point cloud: {len(rack_pcd.points)} points")

# Get bounding box of observed rack
rack_bbox = rack_pcd.get_axis_aligned_bounding_box()
rack_bbox.color = (1, 0, 0)  # Red
rack_extent = rack_bbox.get_extent()
print(f"Rack bounding box extent (x, y, z): {rack_extent}")
print(f"Rack center: {rack_bbox.get_center()}")

# Load CAD model
rack_model_path = project_root / "outputs" / "meshes" / "mug_and_zip" / "040_rack" / "0" / "040_rack_0.stl"
rack_mesh = o3d.io.read_triangle_mesh(str(rack_model_path))
rack_mesh.compute_vertex_normals()

# Get CAD model dimensions
cad_bbox = rack_mesh.get_axis_aligned_bounding_box()
cad_extent = cad_bbox.get_extent()
print(f"\nCAD model bounding box extent (x, y, z): {cad_extent}")

# Calculate suggested scale factor
suggested_scale = rack_extent / cad_extent
print(f"\nSuggested scale factors (x, y, z): {suggested_scale}")
print(f"Average scale factor: {np.mean(suggested_scale):.4f}")

# Apply suggested scale and visualize
rack_mesh_scaled = o3d.geometry.TriangleMesh(rack_mesh)
avg_scale = np.mean(suggested_scale)
rack_mesh_scaled.scale(avg_scale, center=rack_mesh_scaled.get_center())

# Sample points from scaled mesh
rack_model_pcd = rack_mesh_scaled.sample_points_uniformly(number_of_points=10000)
rack_model_pcd.paint_uniform_color([0, 0, 1])  # Blue for model

# Color observed points green
rack_pcd.paint_uniform_color([0, 1, 0])  # Green for observed

# Create coordinate frame at origin
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

print("\nVisualizing:")
print("- GREEN: Observed rack point cloud from RGBD")
print("- BLUE: Scaled CAD model")
print("- RGB axes: Coordinate frame at origin")

o3d.visualization.draw_geometries([rack_pcd, rack_model_pcd, coord_frame, rack_bbox],
                                   window_name="Rack Scale Comparison",
                                   width=1024, height=768)

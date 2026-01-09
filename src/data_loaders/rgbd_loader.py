"""
RGB-D Data Loader for Multi-View Camera System

This module provides a clean interface to load RGB-D images, camera parameters,
and robot trajectories from the recorded dataset.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import cv2


class RGBDDataLoader:
    """
    Loads multi-view RGB-D data with camera calibration.

    This class handles:
    - Loading RGB and depth images from multiple cameras
    - Reading camera intrinsics (focal length, principal point)
    - Reading camera extrinsics (position and orientation in world frame)
    - Loading robot end-effector trajectories

    Attributes:
        data_root: Path to the data directory
        camera_names: List of available camera names (e.g., ['front', 'wrist'])
        num_frames: Total number of frames in the recording
    """

    def __init__(self, data_root: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_root: Path to the data directory containing recording/ and calibration/
        """
        self.data_root = Path(data_root)
        self.recording_dir = self.data_root / "recording"
        self.calibration_dir = self.data_root / "calibration"

        # Discover available cameras
        self.camera_names = self._discover_cameras()

        # Load calibration data (intrinsics and extrinsics)
        self.intrinsics = self._load_intrinsics()
        self.extrinsics = self._load_extrinsics()

        # Count number of frames
        self.num_frames = self._count_frames()

        print(f"[RGBDDataLoader] Initialized with:")
        print(f"  cameras: {self.camera_names}")
        print(f"  frames: {self.num_frames}")

    def _discover_cameras(self) -> list:
        """Find all available camera directories."""
        camera_dirs = []
        for path in self.recording_dir.iterdir():
            if path.is_dir() and path.name.startswith("camera_"):
                # Extract camera name (e.g., "front" from "camera_front")
                camera_name = path.name.replace("camera_", "")
                camera_dirs.append(camera_name)
        return sorted(camera_dirs)

    def _load_intrinsics(self) -> Dict:
        """Load camera intrinsic parameters (focal length, principal point)."""
        intrinsics_path = self.calibration_dir / "intrinsics.json"
        with open(intrinsics_path, 'r') as f:
            return json.load(f)

    def _load_extrinsics(self) -> Dict:
        """Load camera extrinsic parameters (world-to-camera transform)."""
        extrinsics_path = self.calibration_dir / "extrinsics.json"
        with open(extrinsics_path, 'r') as f:
            return json.load(f)

    def _count_frames(self) -> int:
        """Count number of frames by checking RGB images in first camera."""
        first_camera = self.camera_names[0]
        rgb_dir = self.recording_dir / f"camera_{first_camera}" / "rgb"
        return len(list(rgb_dir.glob("*.jpg")))

    def get_rgb(self, camera_name: str, frame_idx: int) -> np.ndarray:
        """
        Load RGB image for a specific camera and frame.

        Args:
            camera_name: Name of the camera (e.g., 'front', 'wrist')
            frame_idx: Frame index (0 to num_frames-1)

        Returns:
            RGB image as numpy array with shape (H, W, 3) and dtype float32 in [0, 1]
        """
        rgb_path = (self.recording_dir / f"camera_{camera_name}" / "rgb"
                    / f"{frame_idx:06d}.jpg")

        # Read image (OpenCV loads as BGR)
        img_bgr = cv2.imread(str(rgb_path))

        # Convert to RGB and normalize to [0, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0

        return img_rgb

    def get_depth(self, camera_name: str, frame_idx: int) -> np.ndarray:
        """
        Load depth image for a specific camera and frame.

        Args:
            camera_name: Name of the camera (e.g., 'front', 'wrist')
            frame_idx: Frame index (0 to num_frames-1)

        Returns:
            Depth image as numpy array with shape (H, W) in meters
        """
        depth_path = (self.recording_dir / f"camera_{camera_name}" / "depth"
                      / f"{frame_idx:06d}.png")

        # Read depth image
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        # Convert to meters (different cameras have different units)
        if camera_name == 'front':
            depth = depth / 1000.0  # mm to m
        elif camera_name == 'wrist':
            depth = depth / 10000.0  # 0.1mm to m

        return depth

    def get_camera_intrinsics(self, camera_name: str) -> np.ndarray:
        """
        Get camera intrinsics matrix K.

        The intrinsics matrix relates 3D points in camera frame to 2D image pixels:
        [u]   [fx  0  cx]   [X]
        [v] = [ 0 fy  cy] * [Y]
        [1]   [ 0  0   1]   [Z]

        Args:
            camera_name: Name of the camera

        Returns:
            3x3 intrinsics matrix K
        """
        params = self.intrinsics[camera_name]
        fx, fy = params['fx'], params['fy']
        cx, cy = params['cx'], params['cy']

        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float32)

        return K

    def get_camera_extrinsics(self, camera_name: str, frame_idx: Optional[int] = None) -> np.ndarray:
        """
        Get camera extrinsics (world-to-camera transform).

        For fixed cameras, this is constant.
        For wrist camera, this changes with robot motion.

        Args:
            camera_name: Name of the camera
            frame_idx: Frame index (required for wrist camera)

        Returns:
            4x4 transformation matrix from world to camera (world_T_camera)
        """
        if camera_name != 'wrist':
            world_T_camera = np.array(self.extrinsics[camera_name], dtype=np.float32)
            return world_T_camera

        if frame_idx is None:
            raise ValueError("frame_idx is required for wrist camera")

        robot_state = self.get_robot_state(frame_idx)
        ee_pos = np.array(robot_state['obs.ee_pos'])
        ee_quat = np.array(robot_state['obs.ee_quat'])

        from utils.transforms import pose_to_transform_matrix
        ee_T_world = pose_to_transform_matrix(ee_pos, ee_quat)
        camera_T_ee = np.array(self.extrinsics[camera_name], dtype=np.float32)

        # V3: Best empirical result (0.2971m centroid distance)
        world_T_camera = ee_T_world @ camera_T_ee

        return world_T_camera

    def get_robot_state(self, frame_idx: int) -> Dict:
        """
        Get robot end-effector state at a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Dictionary with 'obs.ee_pos' (position) and 'obs.ee_quat' (orientation)
        """
        robot_path = self.recording_dir / "robot" / f"{frame_idx:06d}.json"
        with open(robot_path, 'r') as f:
            return json.load(f)

    def get_frame_data(self, frame_idx: int, camera_name: Optional[str] = None) -> Dict:
        """
        Get all data for a specific frame.

        Args:
            frame_idx: Frame index
            camera_name: If specified, only load this camera. Otherwise load all.

        Returns:
            Dictionary containing RGB, depth, intrinsics, extrinsics for each camera,
            plus robot state.
        """
        cameras_to_load = [camera_name] if camera_name else self.camera_names

        data = {
            'frame_idx': frame_idx,
            'cameras': {},
            'robot': self.get_robot_state(frame_idx)
        }

        for cam in cameras_to_load:
            data['cameras'][cam] = {
                'rgb': self.get_rgb(cam, frame_idx),
                'depth': self.get_depth(cam, frame_idx),
                'intrinsics': self.get_camera_intrinsics(cam),
                'extrinsics': self.get_camera_extrinsics(cam, frame_idx)
            }

        return data


if __name__ == "__main__":
    # Example usage
    loader = RGBDDataLoader(data_root="data")

    # Load frame 0
    frame_data = loader.get_frame_data(0)

    print(f"\nFrame 0 data:")
    for cam_name, cam_data in frame_data['cameras'].items():
        print(f"  {cam_name}:")
        print(f"    RGB shape: {cam_data['rgb'].shape}")
        print(f"    Depth shape: {cam_data['depth'].shape}")
        print(f"    Intrinsics K:\n{cam_data['intrinsics']}")

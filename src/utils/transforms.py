import numpy as np 
from typing import Tuple
from scipy.spatial.transform import Rotation

def pose_to_transform_matrix(
        position: np.ndarray,
        quaternion: np.ndarray, 
) -> np.ndarray:
    
    # use scipy to convert quaternion into a rotation 
    rotation = Rotation.from_quat(quaternion)
    R = rotation.as_matrix()

    # build the 4x4 transform matrix 
    T = np.eye(4)
    T[:3, :3] = R  # rotation component 
    T[:3, 3] = position # translation 

    return T 

def transform_to_pose(
        T: np.ndarray      
) -> Tuple[np.ndarray, np.ndarray]:

    position = T[:3, 3]

    R = T[:3, :3]
    rotation = Rotation.from_matrix(R)
    quaternion = rotation.as_quat() # returns [x, y, z, w]

    return position, quaternion
    



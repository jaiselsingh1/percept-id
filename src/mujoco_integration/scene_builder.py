import numpy as np 
from pathlib import Path 
from typing import List, Tuple, Dict 
# this is a XML tree structure within python
import xml.etree.ElementTree as ET 
from xml.dom import minidom 

class MuJoCoSceneBuilder:
    """build a mujoco scene XML from an estimated object pose"""

    def __init__(self, output_path: str = "outputs/scene.xml"):
        self.output_path = Path(output_path)
        self.objects = []

    def add_object(
            self,
            name: str, 
            mesh_path: str, 
            position: np.ndarray, 
            quaternion: np.ndarray, 
            scale: float = 1.0
    ):
        self.objects.append({
            "name" : name, 
            "mesh_path" : str(mesh_path), 
            "position" : position, 
            "quaternion" : quaternion, 
            "scale" : scale
        })

    def build_scene(self) -> str:
        root = ET.Element("mujoco", model = "reconstructed_scene")

        # compiler settings 
        compiler = ET.SubElement(root, "compiler", angle = "radian", meshdir = "meshes")

        # assets (mesh definitions)
        asset = ET.SubElement(root, "asset")
        for obj in self.objects:
            mesh_name = f"{obj['name']}_mesh"
            mesh_file = Path(obj['mesh_path']).name
            ET.SubElement(asset, 'mesh', name=mesh_name, file=mesh_file, scale=f"{obj['scale']} {obj['scale']} {obj['scale']}")
        
        # world body 
        worldbody = ET.SubElement(root, 'worldbody')
        # add lighting 
        ET.SubElement(worldbody, 'light', pos = '0 0 3', dir = '0 0 -1', diffuse = '0.8 0.8 0.8')  
        # add ground plane 
        ET.SubElement(worldbody, 'geom', name="floor", type="plane", size="2 2 0.1", rgba="0.8 0.8 0.8 1")

        # add objects 
        for obj in self.objects:
            pos_str = f"{obj['position'][0]} {obj['position'][1]} {obj['position'][2]}"
            quat_str = f"{obj['quaternion'][3]} {obj['quaternion'][0]} {obj['quaternion'][1]} {obj['quaternion'][2]}"
            body = ET.SubElement(worldbody, 'body', name=obj['name'], pos=pos_str, quat=quat_str)
            ET.SubElement(body, 'geom', type="mesh", mesh=f"{obj['name']}_mesh", rgba="0.8 0.2 0.2 1")
            ET.SubElement(body, 'freejoint')
        
        # convert to pretty XMl string 
        xml_str = ET.tostring(root, encoding = 'unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")

        # remove the extra blank lines that might exist 
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def save_scene(self):
        # save the scene to a XML file 
        self.output_path.parent.mkdir(parents = True, exist_ok = True)
        xml_content = self.build_scene()

        with open(self.output_path, 'w') as f:
            f.write(xml_content)

        print(f"saved mujoco scene to: {self.output_path}")

    def get_meshdir(self) -> Path:
        return self.output_path.parent / "meshes"
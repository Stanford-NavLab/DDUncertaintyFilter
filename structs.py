from collections import namedtuple
import os, sys
import numpy as np
import pandas as pd

Config = namedtuple("Config", "state_dim observation_dim control_dim")

CameraConfig = namedtuple("CameraConfig", "K T_b_c")

GNSSState = namedtuple("GNSSState", "x y z b")

GNSSData = namedtuple("GNSSData", "observations controls gt satellite_states N_observations")

CarPose = namedtuple("CarPose", "x y theta vx vy omega")

FullPose = namedtuple("FullPose", "x y z theta phi psi vx vy vz omega_theta omega_phi omega_psi")

class PathManager:
    def __init__(self, base):
        self.base = base
        
        full_path = self.full_path()
        image_paths = np.sort(os.listdir(full_path))
        self.image_paths = [os.path.join(full_path, im_path) for im_path in image_paths] 
        
    def full_path(self):
        return self.base

    def get_path(self, idx):
        return self.image_paths[idx]

class KITTIPathManager(PathManager):
    def __init__(self, base, name):
        self.base = base
        self.name = name
        self.exts = {
                        'left': 'image_00/data_rect',
                        'right': 'image_01/data_rect',
                    }
        full_path = self.full_path()
        image_paths = np.sort(os.listdir(full_path))
        self.image_paths = [os.path.join(full_path, im_path) for im_path in image_paths] 
        
    def full_path(self, key='left'):
        return os.path.join(self.base, self.name, self.exts[key])

    def get_path(self, idx):
        return self.image_paths[idx]
    
class HKPathManager(PathManager):
    def __init__(self, base, gt):
        super().__init__(base)
        self.gt = pd.read_csv(gt)
        
        
        
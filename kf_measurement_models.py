import torchfilter as tfilter
import torch
import numpy as np
from utils import *

# (state) -> (observation, observation_noise_covariance)
class MyKFMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, config):
        super().__init__(state_dim=config.state_dim, observation_dim=config.observation_dim)
        self.H = config.H
        self.R = config.R
        
    def forward(self, states):
        N, state_dim = states.shape
        assert self.state_dim == state_dim
        expected_observation = (self.H[None, :, :] @ states[:, :, None]).squeeze(-1)
        
        return expected_observation, self.R.expand((N, self.observation_dim, self.observation_dim))
    
    
class GNSSKFMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, config):
        super().__init__(state_dim=config.state_dim, observation_dim=config.observation_dim)
        self.pr_std = config.pr_std
        self.satXYZb = None
        self.idx_pos_mask = np.zeros(self.state_dim, dtype=bool)
        self.idx_b_mask = np.zeros(self.state_dim, dtype=bool)
        self.idx_pos_mask[config.idx.x] = True
        self.idx_pos_mask[config.idx.y] = True
        self.idx_pos_mask[config.idx.z] = True
        self.idx_b_mask[config.idx.b] = True
        
    def update_sats(self, satXYZb):
        self.satXYZb = satXYZb
        self.observation_dim = len(satXYZb)
        
    def forward(self, states):
        N, state_dim = states.shape
        assert self.state_dim == state_dim
        pos = states[:, self.idx_pos_mask]
        bias = states[:, self.idx_b_mask]
        expected_observation = torch.linalg.norm(self.satXYZb[None, :, :3] - pos[:, None, :], dim=-1) + torch.abs(self.satXYZb[None, :, 3] - bias)
        R = torch.eye(self.observation_dim).expand((N, self.observation_dim, self.observation_dim)) * self.pr_std
        
        return expected_observation, R
    
class GNSSDDKFMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, base_pos, N_dim=31+32):
        super().__init__(state_dim=14 + N_dim, observation_dim=10)
        self.pr_std = 5
        self.carr_std = 5
        self.satXYZb = None
        self.idx_code_mask = None
        self.idx_carr_mask = None                     
        self.base_pos = torch.tensor(base_pos) 
         
    def update_sats(self, satXYZb, idx_code_mask, idx_carr_mask, ref_idx, inter_const_bias=None):
        self.satXYZb = satXYZb
        self.idx_code_mask = idx_code_mask
        self.idx_carr_mask = idx_carr_mask
        self.code_dim = np.count_nonzero(idx_code_mask)
        self.carr_dim = np.count_nonzero(idx_carr_mask)
        self.observation_dim = self.code_dim + self.carr_dim
        self.ref_idx = ref_idx
        if inter_const_bias is not None:
            self.inter_const_bias = inter_const_bias
        else:
            self.inter_const_bias = torch.zeros(satXYZb.shape[0])
        
    def forward(self, states):
        N, state_dim = states.shape
        pos = states[:, :3]
        N_allsvs = states[:, 14:]

        expected_observation_code, expected_observation_carr = expected_d_diff(self.satXYZb, states, self.base_pos[None, :], idx_code_mask=self.idx_code_mask, idx_carr_mask=self.idx_carr_mask, ref_idx=self.ref_idx, inter_const_bias=self.inter_const_bias, N_allsvs=N_allsvs)

        expected_observation = torch.cat((expected_observation_code, expected_observation_carr), -1)
        
        R = torch.eye(self.observation_dim)
        R[:self.code_dim, :self.code_dim] *= self.pr_std
        R[self.code_dim:, self.code_dim:] *= self.carr_std
        R = R.expand((N, self.observation_dim, self.observation_dim))
        
        return expected_observation.float(), R.float()
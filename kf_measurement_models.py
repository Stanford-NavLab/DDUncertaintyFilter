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
    def __init__(self, base_pos, N_dim=0, prange_std=5.0, carrier_noN_std=5.0, carrier_N_std=0.5):
        super().__init__(state_dim=10 + N_dim, observation_dim=10)
        self.prange_std = prange_std
        self.carrier_noN_std = carrier_noN_std
        self.carrier_N_std = carrier_N_std
#         self.measurement_std = torch.nn.Parameter(torch.tensor([prange_std, carrier_noN_std]))
        self.measurement_std = torch.tensor([prange_std, carrier_noN_std])
        self.satXYZb = None
        self.idx_code_mask = None
        self.idx_carr_mask = None                     
        self.base_pos = torch.tensor(base_pos) 
         
    def update_sats(self, satXYZb, idx_code_mask, idx_carr_mask, ref_idx, inter_const_bias=None, N_hypo_dict=None, prange_std=None, carrier_noN_std=None, carrier_N_std=None):
        self.satXYZb = satXYZb
        self.idx_code_mask = idx_code_mask
        self.idx_carr_mask = idx_carr_mask
        self.code_dim = np.count_nonzero(idx_code_mask)
        self.carr_dim = np.count_nonzero(idx_carr_mask)
        self.observation_dim = self.code_dim + self.carr_dim
        self.ref_idx = ref_idx
        
        self.N_allsvs_dd = None
        if N_hypo_dict is not None:
            self.N_allsvs_dd = torch.zeros(31+32)
            for key in N_hypo_dict.keys():
                self.N_allsvs_dd[key] = N_hypo_dict[key][0]
            
        if inter_const_bias is not None:
            self.inter_const_bias = inter_const_bias
        else:
            self.inter_const_bias = torch.zeros(satXYZb.shape[0])
            
        if prange_std is not None:
            self.prange_std = prange_std
            
        if carrier_noN_std is not None:
            self.carrier_noN_std = carrier_noN_std
            
        if carrier_N_std is not None:
            self.carrier_N_std = carrier_N_std
        
        
    def forward(self, states):
        N, state_dim = states.shape
        pos = states[:, :3]
        if state_dim>10:
            N_allsvs = states[:, 10:]
            carrier_std = self.carrier_N_std
        else:
            N_allsvs = self.N_allsvs_dd
            carrier_std = self.carrier_noN_std
            if self.N_allsvs_dd is not None:
                N_allsvs = N_allsvs.reshape(1, -1).expand(N, 31+32)
#                 self.measurement_std[1] = self.carrier_N_std
#                 print(self.N_allsvs_dd)

        expected_observation_code, expected_observation_carr = expected_d_diff(self.satXYZb, states, self.base_pos[None, :], idx_code_mask=self.idx_code_mask, idx_carr_mask=self.idx_carr_mask, ref_idx=self.ref_idx, inter_const_bias=self.inter_const_bias, N_allsvs=N_allsvs)

        expected_observation = torch.cat((expected_observation_code, expected_observation_carr), -1)
        
        R = torch.ones(self.observation_dim)
        R[:self.code_dim] = self.prange_std[self.idx_code_mask]
        R[self.code_dim:] = carrier_std[self.idx_carr_mask]
        R = torch.diag(R)
        R = R.expand((N, self.observation_dim, self.observation_dim))
        
        return expected_observation.float(), R.float()
    
class IMUMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, N_dim=0):
        super().__init__(state_dim=10 + N_dim, observation_dim=4)
        self.AHRS_cov = torch.eye(4)
         
    def update_std(self, AHRS_cov):
        self.AHRS_cov = 0.01*torch.eye(4)
        self.AHRS_cov[1:, 1:] = 0.01*AHRS_cov
        
    def forward(self, states):
        N, state_dim = states.shape
        quat = states[:, 3:7]
        
        orientation = tf.quaternion_invert(quat)
        
        R = self.AHRS_cov.expand((N, self.observation_dim, self.observation_dim))
        
        return orientation.float(), R.float()
    
class VOMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, N_dim=0):
        super().__init__(state_dim=10 + N_dim, observation_dim=3)
        self.VO_std = 5
         
    def update_std(self, std):
        self.VO_std = std
        
    def forward(self, states):
        N, state_dim = states.shape
        
        quat = states[:, 3:7].detach()
        quat = torch.div(quat, torch.norm(quat, dim=1)[:, None])
        
        vel_enu = torch.cat((-states[:, 7:8], states[:, 8:10]), dim=-1)
        
        orientation = tf.quaternion_invert(quat)
        body_velocity = tf.quaternion_apply(orientation, vel_enu)
#         print(vel_enu[0].detach().numpy(), body_velocity[0].detach().numpy(), orientation[0].detach().numpy())
        
        R = torch.diag(torch.tensor([1e-5, self.VO_std, 1e-5])).expand((N, self.observation_dim, self.observation_dim))
        
        return body_velocity.float(), R.float()
    
class IMUDDMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, *args, N_dim=0, **kwargs):
        super().__init__(state_dim=10 + N_dim, observation_dim=14)
        self.imu_model = IMUMeasurementModel(N_dim=N_dim)
        self.gnss_model = GNSSDDKFMeasurementModel(*args, N_dim=N_dim, **kwargs)
        self.mode = "imu"
         
    def update_imu_std(self,  *args, **kwargs):
        self.imu_model.update_std( *args, **kwargs)
        self.observation_dim = 4
        self.mode = "imu"
        
    def update_sats(self,  *args, **kwargs):
        self.gnss_model.update_sats( *args, **kwargs)
        self.observation_dim = self.gnss_model.observation_dim
        self.mode = "gnss"
        
    def forward(self, states):
        N, state_dim = states.shape
        
        if self.mode=="imu":
            meas, R = self.imu_model(states)
        elif self.mode=="gnss":
            meas, R = self.gnss_model(states)
        
        return meas, R
    
class IMU_VO_DD_MeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, *args, N_dim=0, **kwargs):
        super().__init__(state_dim=10 + N_dim, observation_dim=14)
        self.imu_model = IMUMeasurementModel(N_dim=N_dim)
        self.vo_model = VOMeasurementModel(N_dim=N_dim)
        self.gnss_model = GNSSDDKFMeasurementModel(*args, N_dim=N_dim, **kwargs)
        self.mode = "imu"
         
    def update_imu_std(self,  *args, **kwargs):
        self.imu_model.update_std( *args, **kwargs)
        self.observation_dim = self.imu_model.observation_dim
        self.mode = "imu"
        
    def update_vo_std(self,  *args, **kwargs):
        self.vo_model.update_std( *args, **kwargs)
        self.observation_dim = self.vo_model.observation_dim
        self.mode = "vo"
        
    def update_sats(self,  *args, **kwargs):
        self.gnss_model.update_sats( *args, **kwargs)
        self.observation_dim = self.gnss_model.observation_dim
        self.mode = "gnss"
        
    def forward(self, states):
        N, state_dim = states.shape
        
        if self.mode=="imu":
            meas, R = self.imu_model(states)
        elif self.mode=="vo":
            meas, R = self.vo_model(states)
        elif self.mode=="gnss":
            meas, R = self.gnss_model(states)
        
        return meas, R
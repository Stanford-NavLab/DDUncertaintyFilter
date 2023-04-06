import torchfilter as tfilter
import torch
import numpy as np
import torch.autograd.functional as F
from utils import *
from robust_cost_models import *

# (state) -> (observation, observation_noise_covariance)
class SimpleKFMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, config):
        super().__init__(state_dim=config.state_dim, observation_dim=config.observation_dim)
        self.H = config.H
        self.R = config.R
        
    def forward(self, states):
        N, state_dim = states.shape
        assert self.state_dim == state_dim
        expected_observation = (self.H[None, :, :] @ states[:, :, None]).squeeze(-1)
        
        return expected_observation, self.R.expand((N, self.observation_dim, self.observation_dim))
    
# Measurement model written in torchfilter library for GNSS code phase measurements    
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
    
# Measurement model written in torchfilter library for GNSS double difference code and carrier phase measurements
class GNSSDDKFMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, base_pos, N_dim=0, prange_std=5.0, carrier_std=0.1, include_carrier=False, prange_std_tail=None):
        super().__init__(state_dim=16 + N_dim, observation_dim=10)
        self.base_pos = base_pos
        self.prange_std = prange_std
        self.carrier_std = carrier_std
        self.ref_idx = 0
        self.include_carrier = include_carrier
        self.include_correntropy = prange_std_tail is not None
        if self.include_correntropy:
            self.prange_std_tail = prange_std_tail
        self.linearization_point = None
        self.robust_cost_model = MEstimationModel(threshold=20.0)
            

    def update(self, satXYZb=None, ref_idx=None, inter_const_bias=None, idx_code_mask=None, idx_carr_mask=None, prange_std=None, carrier_std=None, estimated_N=None, linearization_point=None):
        if satXYZb is not None:
            self.satXYZb = satXYZb
        if ref_idx is not None:
            self.ref_idx = ref_idx
        if inter_const_bias is not None:
            self.inter_const_bias = inter_const_bias
        else:
            self.inter_const_bias = torch.zeros(self.satXYZb.shape[0])
        if prange_std is not None:
            self.prange_std = prange_std
        if carrier_std is not None:
            self.carrier_std = carrier_std
        if estimated_N is not None:
            self.estimated_N = estimated_N
        else:
            self.estimated_N = torch.zeros(1, self.satXYZb.shape[0])
        if idx_code_mask is not None:
            self.idx_code_mask = idx_code_mask
        else:
            self.idx_code_mask = torch.ones(self.satXYZb.shape[0], dtype=torch.bool)
        if idx_carr_mask is not None:
            self.idx_carr_mask = idx_carr_mask
        else:
            self.idx_carr_mask = torch.ones(self.satXYZb.shape[0], dtype=torch.bool)
        self.code_dim = torch.sum(self.idx_code_mask)
        self.carr_dim = torch.sum(self.idx_carr_mask)
        if self.include_carrier:
            self.observation_dim = self.code_dim + self.carr_dim
        else:
            self.observation_dim = self.code_dim
        if linearization_point is not None:
            self.linearization_point = linearization_point
        
    def robust_cost_cholesky(self, observation, expected_observation):
        with torch.no_grad():
            residual = observation - expected_observation
            residual[:, :self.code_dim] = residual[:, :self.code_dim]/self.prange_std
            residual[:, self.code_dim:] = residual[:, self.code_dim:]/self.carrier_std
            outlier_mask = self.robust_cost_model.generate_outlier_mask(residual)
            
        N, _ = residual.shape
        R = torch.ones(N, self.observation_dim)
        R[:, :self.code_dim] = self.prange_std
        R[:, self.code_dim:] = self.carrier_std
        R[outlier_mask] = 1e6
        R = torch.diag_embed(R)

        return R


    def forward(self, states):
        N, state_dim = states.shape
        self.state_dim = state_dim

        pos = states[:, 0:3]
        expected_observation_code, expected_observation_carr = expected_d_diff(self.satXYZb, states, self.base_pos[None, :], idx_code_mask=self.idx_code_mask, idx_carr_mask=self.idx_carr_mask, ref_idx=self.ref_idx, inter_const_bias=self.inter_const_bias, N_allsvs=self.estimated_N)

        if self.include_carrier:
            expected_observation = torch.cat((expected_observation_code, expected_observation_carr), dim=1)
        else:
            expected_observation = expected_observation_code

        # Calculate covariance matrix
        if self.linearization_point is None:
            self.linearization_point = expected_observation
        R = self.robust_cost_cholesky(self.linearization_point, expected_observation)

        return expected_observation.float(), R.float()
    
class IMUMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, N_dim=0):
        super().__init__(state_dim=16 + N_dim, observation_dim=4)
        self.r_std = torch.tensor(np.deg2rad(1.0))
        self.p_std = torch.tensor(np.deg2rad(1.0))
        self.y_std = torch.tensor(np.deg2rad(1.0))
        self.linearization_point = None
        self.robust_cost_model = MEstimationModel(threshold=5.0, debug=False)
         
    def update(self, r_std=None, p_std=None, y_std=None, linearization_point=None):
        if r_std is not None:
            self.r_std = r_std
        if p_std is not None:
            self.p_std = p_std
        if y_std is not None:
            self.y_std = y_std
        if linearization_point is not None:
            self.linearization_point = linearization_point

    def cholesky(self, quat):
        # Return the covariance matrix of the noise.
        with torch.enable_grad():
            tmp = quat.detach().clone()
            N, _ = tmp.shape
            tmp = tmp[:, None, :].expand((N, 3, 4))
            tmp.requires_grad = True
            eul = quat2eul(tmp.reshape(-1, 4)).reshape(N, -1, 3)
            mask = torch.eye(3, device=eul.device).repeat(N, 1, 1)
            quat_jacobian = torch.autograd.grad(eul, tmp, mask, create_graph=False)[0].detach().float()
            quat_jacobian = quat_jacobian.transpose(1, 2)
            ret_chol = torch.matmul(quat_jacobian, torch.diag(torch.stack([self.r_std, self.p_std, self.y_std])).float())
            ret_chol = torch.linalg.norm(ret_chol, dim=2)
            ret_chol = torch.diag_embed(ret_chol)
        return ret_chol
    
    def robust_cost_cholesky(self, observation, expected_observation):
        base_R = self.cholesky(expected_observation)
        
        with torch.no_grad():
            residual = observation - expected_observation
            N, _ = residual.shape
            # normalize residual based on covariance base_R
            residual = (residual[:, None, :] @ torch.linalg.inv(base_R)).reshape(N, -1)

            outlier_mask = self.robust_cost_model.generate_outlier_mask(residual)
        
        R = torch.diagonal(base_R, dim1=-2, dim2=-1)
        R[outlier_mask] = 10
        R = torch.diag_embed(R)
        
        return R

    # def cholesky(self):
    #     # Return the cholesky matrix of the noise.
    #     return torch.diag(torch.stack([(self.r_std + self.p_std + self.y_std)/3, self.r_std, self.p_std, self.y_std])).float()
        
    def forward(self, states):
        N, state_dim = states.shape
        self.state_dim = state_dim

        quat = states[:, 6:10]
        
#         quat_hat = torch.mean(quat.detach().clone(), dim=0, keepdim=False)
#         quat_hat = quat_hat / quat_hat.norm()
        
#         print("Meas ", self.covariance(quat_hat))
        
        # R = self.robust_cost_cholesky(self.linearization_point, quat)
        R = self.cholesky(quat)
        # R = self.cholesky().expand((N, 4, 4))

        
        return quat.float(), R.float()
    
class VOMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, N_dim=0):
        super().__init__(state_dim=16 + N_dim, observation_dim=3)
        self.std = torch.tensor(5.0)
        self.scale = torch.tensor(1.0)
         
    def update(self, std=None, scale=None):
        if std is not None:
            self.std = std
        if scale is not None:
            self.scale = scale
        
    def forward(self, states):
        N, state_dim = states.shape
        
        quat = states[:, 6:10].detach().clone()
        quat = torch.div(quat, torch.norm(quat, dim=1)[:, None])
        
        vel_enu = states[:, 3:6]*self.scale
        
        body_velocity = tf.quaternion_apply(quat, vel_enu)
#         print(vel_enu[0].detach().numpy(), body_velocity[0].detach().numpy(), orientation[0].detach().numpy())
        
        R = torch.diag(torch.stack([torch.tensor(1e-2), self.std, torch.tensor(1e-2)])).expand((N, self.observation_dim, self.observation_dim))
        
        return body_velocity.float(), R.float()
    
class VOLandmarkMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, N_dim=0):
        super().__init__(state_dim=16 + N_dim, observation_dim=1)
        # Set measurement noise covariance
        self.landmark_std = torch.tensor(10.0)   # N_landmarks*2
        self.scale = torch.tensor(1.0)
        self.linearization_point = None
        self.robust_cost_model = MEstimationModel(threshold=10.0, debug=False)

    def update(self, landmark_std=None, landmarks=None, intrinsic=None, scale=None, prev_state=None, linearization_point=None):
        if landmark_std is not None:
            self.landmark_std = landmark_std
        if landmarks is not None:
            self.landmarks = landmarks
        if intrinsic is not None:
            self.K = intrinsic
        if scale is not None:
            self.scale = scale
        if prev_state is not None:
            self.prev_state = prev_state
        if linearization_point is not None:
            self.linearization_point = linearization_point
        self.observation_dim = 2*self.landmarks.shape[0]
        
    
    def robust_cost_cholesky(self, observation, expected_observation):
        with torch.no_grad():
            residual = observation - expected_observation
            residual = residual/self.landmark_std
            outlier_mask = self.robust_cost_model.generate_outlier_mask(residual)
            
        N, _ = residual.shape
        R = torch.ones(N, self.observation_dim)*self.landmark_std
        R[outlier_mask] = 1e6
        R = torch.diag_embed(R)

        return R
    
    def forward(self, states):
        # Generate expected 2d landmark locations from the current state and 3d landmark locations
        # Input: states: N*state_dim
        # Output: expected 2d landmark x coordinate: N*N_landmarks
        #         expected 2d landmark y coordinate: N*N_landmarks
        #        expected 2d landmark x covariance: N*N_landmarks*N_landmarks
        #        expected 2d landmark y covariance: N*N_landmarks*N_landmarks
        N, state_dim = states.shape
        self.state_dim = state_dim
        
        # Extract the orientation quaternion from state variables
        quat = states[:, 6:10].detach().clone()
        
        # Normalize the quaternion (world -> body)
        orientation = torch.div(quat, torch.norm(quat, dim=1)[:, None])
        # Retrieve the landmark positions in previous frame
        landmarks = self.landmarks  # N_landmarks*3
        
        # Previous quaternion
        prev_quat = self.prev_state[:, 6:10].detach().clone()
        
        # Compute the change in rotation between frames (TODO: unit test this)
        # delta_quat = quat_delta(prev_quat, quat)  # N*3
        delta_quat = quat_delta(quat, quat)
        # relative rotation matrix
        rmat = tf.quaternion_to_matrix(delta_quat).float().expand(N, 3, 3)
        # Compute the ENU motion from state variables
        vel_enu = (states[:, :3]-self.prev_state[:, :3])*self.scale
        # Transform the ENU velocity to body frame in camera reference
        body_velocity = tf.quaternion_apply(orientation, vel_enu)
        body_velocity = body_velocity[:, [0, 2, 1]].reshape(N, 3, 1)
        body_velocity[:, 2, :] = -body_velocity[:, 2, :]
        # body_velocity[:, :2, :] = 0.0
        
        extr = torch.cat([rmat, body_velocity], dim=-1)
        # print(extr[0].detach().numpy())
        extr = torch.cat([extr, torch.tensor([[0, 0, 0, 1]]).expand(N, 1, 4)], dim=1)
        
        # Calculate intrinsic matrix
        intr = torch.cat([self.K, torch.zeros(3, 1)], dim=1)
        intr = torch.cat([intr, torch.tensor([[0, 0, 0, 1]])], dim=0)
        # Calculate projection matrix
        proj = torch.bmm(intr.expand(N, 4, 4), extr)   # N*4*4
        
        # Convert 3D points to homogeneous coordinates
        obj_pts = torch.cat([self.landmarks, torch.ones(self.landmarks.shape[0], 1)], dim=1).expand(N, self.landmarks.shape[0], 4)
        # Project 3D points to image plane
        img_pts = torch.bmm(proj, obj_pts.transpose(2, 1))
        img_pts = img_pts[:, :2, :] / img_pts[:, 2:3, :]
        img_pts = img_pts.transpose(2, 1).reshape(N, -1)  # N*(N_landmarks*2)
        
        # Calculate covariance of expected 2d landmark locations
        R = self.robust_cost_cholesky(self.linearization_point, img_pts)   # N*(N_landmarks*2)*(N_landmarks*2)

        return img_pts.float(), R.float()
    
class IMUDDMeasurementModel(tfilter.base.KalmanFilterMeasurementModel):
    def __init__(self, *args, N_dim=0, **kwargs):
        super().__init__(state_dim=16 + N_dim, observation_dim=14)
        self.imu_model = IMUMeasurementModel(N_dim=N_dim)
        self.gnss_model = GNSSDDKFMeasurementModel(*args, N_dim=N_dim, **kwargs)
        self.mode = "imu"
         
    def update_imu(self,  *args, **kwargs):
        self.imu_model.update( *args, **kwargs)
        self.observation_dim = self.imu_model.observation_dim
        self.mode = "imu"
        
    def update_gnss(self,  *args, **kwargs):
        self.gnss_model.update( *args, **kwargs)
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
        super().__init__(state_dim=16 + N_dim, observation_dim=14)
        self.imu_model = IMUMeasurementModel(N_dim=N_dim)
        self.vo_base_model = VOMeasurementModel(N_dim=N_dim)
        self.vo_model = VOLandmarkMeasurementModel(N_dim=N_dim)
        self.gnss_model = GNSSDDKFMeasurementModel(*args, N_dim=N_dim, **kwargs)
        self.mode = "imu"
         
    def update_imu(self,  *args, **kwargs):
        self.imu_model.update( *args, **kwargs)
        self.observation_dim = self.imu_model.observation_dim
        self.state_dim = self.imu_model.state_dim
        self.mode = "imu"
        
    def update_gnss(self,  *args, **kwargs):
        self.gnss_model.update( *args, **kwargs)
        self.observation_dim = self.gnss_model.observation_dim
        self.state_dim = self.gnss_model.state_dim
        self.mode = "gnss"
        
    def update_vo(self,  *args, **kwargs):
        self.vo_model.update( *args, **kwargs)
        self.observation_dim = self.vo_model.observation_dim
        self.state_dim = self.vo_model.state_dim
        self.mode = "vo"
        
    def update_vo_base(self,  *args, **kwargs):
        self.vo_base_model.update( *args, **kwargs)
        self.observation_dim = self.vo_base_model.observation_dim
        self.state_dim = self.vo_base_model.state_dim
        self.mode = "vo_base"
        
    def forward(self, states):
        N, state_dim = states.shape
        
        if self.mode=="imu":
            meas, R = self.imu_model(states)
        elif self.mode=="vo":
            meas, R = self.vo_model(states)
        elif self.mode=="vo_base":
            meas, R = self.vo_base_model(states)
        elif self.mode=="gnss":
            meas, R = self.gnss_model(states)
        
        return meas, R
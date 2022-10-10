import torchfilter as tfilter
import torch
import numpy as np
import pytorch3d.transforms as tf
from utils import *

# (state, control) -> (state, propagation_noise_covariance)
class MyDynamicsModel(tfilter.base.DynamicsModel):
    def __init__(self, config):
        super().__init__(state_dim=config.state_dim)
        self.A = config.A
        self.B = config.B
        self.Q = config.Q
        
    def forward(self, initial_states, controls):
        N, state_dim = initial_states.shape
        assert self.state_dim == state_dim
        predicted_states = batched_mm(self.A, initial_states) + batched_mm(self.B, controls) 
        
        return predicted_states, self.Q.expand((N, state_dim, state_dim))
    
    
class FullPoseDynamicsModel(tfilter.base.DynamicsModel):
    def __init__(self):
        super().__init__(state_dim=12)
        self.A = torch.eye(self.state_dim)
        self.A[:6, 6:] = torch.eye(6)*0.1 
        self.B = torch.zeros(12, 3)
        self.B[6, 0] = 1
        self.B[7, 1] = 1
        self.B[9, 2] = 1
        prop_std = torch.ones(12)
        prop_std[:6] *= 0.3
        prop_std[6:8] *= 0.05
        prop_std[8] *= 1e-5
        prop_std[9] *= (np.pi/180)*5
        prop_std[10:] *= 1e-5
        self.Q = torch.diag(prop_std)
        
    def forward(self, initial_states, controls):
        N, state_dim = initial_states.shape
        assert self.state_dim == state_dim
        predicted_states = batched_mm(self.A, initial_states) + batched_mm(self.B, controls) 
        
        return predicted_states, self.Q.expand((N, state_dim, state_dim))
    

    
class CarPoseDynamicsModel(tfilter.base.DynamicsModel):
    def __init__(self):
        super().__init__(state_dim=6)
        self.A = torch.eye(self.state_dim)
        self.A[:3, 3:] = torch.eye(3)*0.1 
        self.B = torch.zeros(6, 3)
        self.B[3:, :] = torch.eye(3)
        prop_std = torch.ones(6)
        prop_std[:2] *= 0.3   # x, y
        prop_std[2] *= (np.pi/180)*5  # theta
        prop_std[3:5] *= 0.05    # vx, vy
        prop_std[5] *= (np.pi/180)*0.5  # omega
        self.Q = torch.diag(prop_std)
        
    def forward(self, initial_states, controls):
        N, state_dim = initial_states.shape
        assert self.state_dim == state_dim
        predicted_states = batched_mm(self.A, initial_states) + batched_mm(self.B, controls) 
        
        return predicted_states, self.Q.expand((N, state_dim, state_dim))
    

# DYNAMICS MODEL

class CarFullPoseDynamicsModel(tfilter.base.DynamicsModel):
    def __init__(self, N_dim=31+32, dt = 0.0025):
        super().__init__(state_dim=10 + N_dim)
        self.N_dim = N_dim
        self.xy_std = torch.nn.Parameter(torch.tensor(10.0))
        self.z_std = torch.nn.Parameter(torch.tensor(0.005))
        self.q_std = torch.tensor([0.01, 0.01, 0.01, 0.01])
        self.v_std = torch.tensor([1.0, 1.0, 0.1])
        
        self.Qxy = torch.eye(2)*self.xy_std
        self.Qz = self.z_std
        self.Qq = torch.diag(self.q_std)
        self.Qv = torch.diag(self.v_std)
        
        if self.N_dim > 0:
            self.QN = torch.eye(self.N_dim)  # N
        
        self.update_dt_cov(dt, self.Qq[:3, :3], self.Qv)
        
#         self.save_data = {
#             'a_enu': [],
#             'q_dot': []
#         }
    
    def update_ambiguity_cov(self, sigma):
        self.QN = sigma
        self.QN = torch.diag(self.QN)
    
    def update_dt_cov(self, dt, q_cov, v_cov):
        self.dt = dt
        self.Qq[:3, :3] = q_cov
        self.Qv = v_cov
        
    def calc_Q(self, dim=None):
        Q = torch.eye(self.state_dim)
        Q[:2, :2] = self.Qxy
        Q[2, 2] = self.Qz
        Q[3:7, 3:7] = self.Qq
        Q[7:10, 7:10] = self.Qv
        
        if self.N_dim > 0:
            Q[10:, 10:] = self.QN
            
        if dim is not None:
            Q = Q[:dim, :dim]
            
        return Q*self.dt
        
    def forward(self, initial_states, controls):
        N, state_dim = initial_states.shape
        assert self.state_dim == state_dim
        
        x = initial_states[:, :3]
        quat = torch.div(initial_states[:, 3:7], torch.norm(initial_states[:, 3:7], dim=1)[:, None])
        x_dot = initial_states[:, 7:10]
        
        accel = controls[:, :3]
        omega = controls[:, 3:6]
        
        gravity = torch.zeros((N, 3))
        gravity[:, -1] = 9.81
        
        accel_enu = tf.quaternion_apply(quat, accel) - gravity
        accel_enu[:, :] = 0.0
        
#         self.save_data['a_enu'].append(accel_enu[0, :])
        
        quat_dot = compute_quat_dot(quat, omega)
        
#         self.save_data['q_dot'].append(quat_dot[0, :])
        
        predicted_x = x + self.dt*x_dot
        
        _predicted_quat = quat + self.dt*quat_dot
        predicted_quat = torch.div(_predicted_quat, torch.norm(_predicted_quat, dim=1)[:, None])
        
        predicted_x_dot = x_dot + self.dt*accel_enu
        
        predicted_states = torch.cat((predicted_x, predicted_quat, predicted_x_dot), -1)
        
        if self.N_dim > 0:
            N_allsv = initial_states[:, 10:]
            predicted_states = torch.cat((predicted_states, N_allsv), -1)
        
        Q = self.calc_Q()
        
        return predicted_states, Q.expand((N, state_dim, state_dim))
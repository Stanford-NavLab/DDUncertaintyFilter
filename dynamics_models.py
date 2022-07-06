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
    def __init__(self, N_dim=31+32):
        super().__init__(state_dim=(3+4)*2 + N_dim)
        dt = 0.0025
        self.N_dim = N_dim
        self.update_dt(dt)
    
    def update_dt(self, dt):
        fac_a = 0.1 # Retention factor
        fac_b = 0.7 # Weighting factor
        
        self.A = torch.eye(self.state_dim - self.N_dim)
        self.A[3:7, 3:7] = torch.eye(4)*fac_b
        self.A[10:14, 10:14] = torch.eye(4)*fac_a
        self.A[:3, 7:10] = torch.eye(3)*dt
        self.A[3:7, 10:14] = torch.eye(4)*dt*fac_b
        
        self.B = torch.zeros(self.state_dim - self.N_dim, 11)
        self.B[3:7, 7:11] = torch.eye(4)*(1-fac_b)
        self.B[7:10, :3] = torch.eye(3)*dt
        self.B[10:14, 3:7] = torch.eye(4)*(1-fac_a)
        
        prop_std = torch.ones(self.state_dim)
        prop_std[:3] *= np.sqrt(dt*10)   # x, y, z
        prop_std[3:7] *= np.sqrt(dt*0.1)  # q
        prop_std[7:10] *= np.sqrt(dt*1)    # v
        prop_std[10:14] *= np.sqrt(dt*0.1)  # q_dot
        if self.N_dim > 0:
            prop_std[14:] *= 1  # N
        self.Q = torch.diag(prop_std)
    
    def forward(self, initial_states, controls):
        N, state_dim = initial_states.shape
        assert self.state_dim == state_dim
        
        or_quat = initial_states[:, 3:7].detach()
        term_3 = controls[:, 6:]
#         or_quat = term_3
        
        gravity = torch.zeros((N, 3))
        gravity[:, -1] = 9.81
        term_1 = tf.quaternion_apply(or_quat, controls[:, :3]) - gravity
        term_2 = quat_dot(or_quat, controls[:, 3:6])
        
        controls = torch.cat((term_1, term_2, term_3), -1)
        
        x_q_xdot_qdot = initial_states[:, :14]
        N_allsv = initial_states[:, 14:]
        
        predicted_states = batched_mm(self.A, x_q_xdot_qdot) + batched_mm(self.B, controls) 
        
        return torch.cat((predicted_states, N_allsv), -1), self.Q.expand((N, state_dim, state_dim))
import torchfilter as tfilter
import torch
import numpy as np
import pytorch3d.transforms as tf
import torch.autograd.functional as F
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
class XYZModel(tfilter.base.DynamicsModel):
    def __init__(self, dt = 0.0025):
        # Initialise the dynamics model, which is a subclass of the base class.
        # The state dimension is 3, that is, [x, y, z].
        super().__init__(state_dim=3)
        
        # Initialise the standard deviations of the noise for each state dimension.
        self.x_std = torch.tensor(1.0)
        self.y_std = torch.tensor(1.0)
        self.z_std = torch.tensor(0.05)
        self.dt = dt
        
    def update(self, dt=None, x_std=None, y_std=None, z_std=None):
        # Update the standard deviations of the noise for each state dimension.
        if dt is not None:
            self.dt = dt
        if x_std is not None:
            self.x_std = x_std*self.dt
        if y_std is not None:
            self.y_std = y_std*self.dt
        if z_std is not None:
            self.z_std = z_std*self.dt

    def cholesky(self, mode='xyz'):
        # Return the covariance matrix of the noise.
        if mode == 'xyz':
            return torch.diag(torch.stack([self.x_std, self.y_std, self.z_std]))
        elif mode == 'xy':
            return torch.diag(torch.stack([self.x_std, self.y_std]))
        elif mode == 'x':
            return self.x_std
        else:
            raise ValueError('Invalid mode.')

    def forward(self, initial_states, controls):
        # The dynamics model is a simple constant velocity model.
        # The controls are the velocity/acceleration.
        N, state_dim = initial_states.shape
        assert self.state_dim == state_dim

        return initial_states + controls*self.dt, self.cholesky().expand((N, 3, 3))

class QuaternionOrientationModel(tfilter.base.DynamicsModel):
    def __init__(self, dt = 0.0025):
        # Initialise the dynamics model, which is a subclass of the base class.
        # The state dimension is 3, that is, [qw, qx, qy, qz].
        super().__init__(state_dim=4)

        # Initialise the standard deviations of the noise for each state dimension.
        self.r_std = torch.tensor(np.deg2rad(5.0))
        self.p_std = torch.tensor(np.deg2rad(5.0))
        self.y_std = torch.tensor(np.deg2rad(5.0))
        self.dt = dt

    def update(self, dt=None, r_std=None, p_std=None, y_std=None):
        # Update the standard deviations of the noise for each state dimension.
        if dt is not None:
            self.dt = dt
        if r_std is not None:
            self.r_std = r_std*self.dt
        if p_std is not None:
            self.p_std = p_std*self.dt
        if y_std is not None:
            self.y_std = y_std*self.dt

    def cholesky(self, quat):
        # Return the covariance matrix of the noise.
        tmp = quat.detach().clone()
        tmp.requires_grad = True
        quat_jacobian = F.jacobian(quat2eul, tmp).detach().float()
        
        return torch.matmul(quat_jacobian.transpose(0, 1), torch.diag(torch.stack([self.r_std, self.p_std, self.y_std])).float())

    def forward(self, initial_states, controls):
        # The dynamics model is a simple constant angular velocity model.
        # The controls are the angular velocity.
        N, state_dim = initial_states.shape
        assert self.state_dim == state_dim

        quat_dot = compute_quat_dot(initial_states, controls)

        quat = initial_states + quat_dot*self.dt
        quat = quat / quat.norm(dim=1, keepdim=True)

        quat_hat = torch.mean(quat.detach().clone(), dim=0, keepdim=False)
        quat_hat = quat_hat / quat_hat.norm()
        
#         print("Dynamics ", self.covariance(quat_hat))

        return quat, self.cholesky(quat_hat).expand((N, 4, 3))

class AmbiguityModel(tfilter.base.DynamicsModel):
    def __init__(self, state_dim=1, dt = 0.0025):
        # Initialise the dynamics model, which is a subclass of the base class.
        # The state dimension is an input, that is, [N1, ..., Nk].
        super().__init__(state_dim=state_dim)
        
        # Initialise the standard deviations of the noise for each state dimension.
        self.amb_std = torch.tensor(1.0)    # Ambiguity noise
        self.cs_amb_std = torch.tensor(10.0)    # Cycle slip ambiguity noise
        self.dt = dt
        
    def update(self, dt=None, amb_std=None, cs_amb_std=None):
        # Update the standard deviations of the noise for each state dimension.
        if dt is not None:
            self.dt = dt
        if amb_std is not None:
            self.amb_std = amb_std*self.dt
        if cs_amb_std is not None:
            self.cs_amb_std = cs_amb_std*self.dt

    def cholesky(self, cs_mask):
        # Return the covariance matrix of the noise.
        return torch.diag(torch.where(cs_mask, self.cs_amb_std, self.amb_std))
        
    def forward(self, initial_states, controls):
        # The dynamics model is a simple random walk model.
        # The controls are null.
        N, state_dim = initial_states.shape
        self.state_dim = state_dim

        return initial_states, self.cholesky().expand((N, state_dim, state_dim))


class IMUBiasModel(tfilter.base.DynamicsModel):
    def __init__(self, dt = 0.0025):
        # Initialise the dynamics model, which is a subclass of the base class.
        # The state dimension is an input, that is, [N1, ..., Nk].
        super().__init__(state_dim=6)
        
        # Initialise the standard deviations of the noise for each state dimension.
        self.acc_bias_std = torch.tensor(1.0)    # Accelerometer bias noise
        self.gyr_bias_std = torch.tensor(1.0)    # Gyroscope bias noise
        self.dt = dt
        
    def update(self, dt=None, acc_bias_std=None, gyr_bias_std=None):
        # Update the standard deviations of the noise for each state dimension.
        if dt is not None:
            self.dt = dt
        if acc_bias_std is not None:
            self.acc_bias_std = acc_bias_std*self.dt
        if gyr_bias_std is not None:
            self.gyr_bias_std = gyr_bias_std*self.dt

    def cholesky(self):
        # Return the covariance matrix of the noise.
        return torch.diag(torch.stack([self.acc_bias_std, self.acc_bias_std, self.acc_bias_std, self.gyr_bias_std, self.gyr_bias_std, self.gyr_bias_std]))
        
    def forward(self, initial_states, controls):
        # The dynamics model is a simple random walk model.
        # The controls are null.
        N, state_dim = initial_states.shape
        assert self.state_dim == state_dim

        return initial_states, self.cholesky().expand((N, 6, 6))

# Combine previously defined pos and vel models into a single dynamics model
class PosVelModel(tfilter.base.DynamicsModel):
    def __init__(self, dt = 0.0025):
        # Initialise the dynamics model, which is a subclass of the base class.
        # The state dimension is an input.
        super().__init__(state_dim=6)
        
        # Initialise the standard deviations of the noise for each state dimension.
        self.pos_model = XYZModel(dt=dt)
        self.vel_model = XYZModel(dt=dt)
        self.update(pos_x_std=1.0, pos_y_std=1.0, pos_z_std=1.0, vel_x_std=1e-2, vel_y_std=1.0, vel_z_std=1e-2)
        
    def update(self, dt=None, pos_x_std=None, pos_y_std=None, pos_z_std=None, vel_x_std=None, vel_y_std=None, vel_z_std=None):
        # Update the standard deviations of the noise for each state dimension.
        self.pos_model.update(dt=dt, x_std=pos_x_std, y_std=pos_y_std, z_std=pos_z_std)
        self.vel_model.update(dt=dt, x_std=vel_x_std, y_std=vel_y_std, z_std=vel_z_std)

    def forward(self, initial_states, controls):
        # The dynamics model is a simple random walk model.
        # The controls are null.
        N, state_dim = initial_states.shape
        assert self.state_dim == state_dim
        
        pos, pos_cov = self.pos_model(initial_states[:, 0:3], initial_states[:, 3:6])
        vel, vel_cov = self.vel_model(initial_states[:, 3:6], controls)
        
        # Combine the position and velocity states and covariance matrices
        cholseky_matrix = torch.zeros((N, 6, 6))
        cholseky_matrix[:, 0:3, 0:3] = pos_cov
        cholseky_matrix[:, 3:6, 3:6] = vel_cov
        
        return torch.cat((pos, vel), axis=1), cholseky_matrix

# Combine previously defined models into a single dynamics model
class PosVelQuatBiasModel(tfilter.base.DynamicsModel):
    def __init__(self, dt = 0.0025):
        # Initialise the dynamics model, which is a subclass of the base class.
        # The state dimension is an input.
        super().__init__(state_dim=16)
        
        # Initialise the standard deviations of the noise for each state dimension.
        self.pos_vel_model = PosVelModel(dt=dt)
        self.quat_model = QuaternionOrientationModel(dt=dt)
        self.imu_bias_model = IMUBiasModel(dt=dt)
        
    def update(self, dt=None, pos_x_std=None, pos_y_std=None, pos_z_std=None, vel_x_std=None, vel_y_std=None, vel_z_std=None, r_std=None, p_std=None, y_std=None, acc_bias_std=None, gyr_bias_std=None):
        # Update the standard deviations of the noise for each state dimension.
        self.pos_vel_model.update(dt=dt, pos_x_std=pos_x_std, pos_y_std=pos_y_std, pos_z_std=pos_z_std, vel_x_std=vel_x_std, vel_y_std=vel_y_std, vel_z_std=vel_z_std)
        self.quat_model.update(dt=dt, r_std=r_std, p_std=p_std, y_std=y_std)
        self.imu_bias_model.update(dt=dt, acc_bias_std=acc_bias_std, gyr_bias_std=gyr_bias_std)
        
    def forward(self, initial_states, controls):
        # The dynamics model is a simple random walk model.
        # The controls are null.
        N, state_dim = initial_states.shape
        self.state_dim = state_dim

        accel = controls[:, :3] - initial_states[:,10:13]
        accel_mag = torch.linalg.norm(accel, dim=1)
#         print("Accel mag: ", accel_mag)
        accel_body = torch.zeros(N, 3)
#         accel_body[:, 1] = accel_mag
        accel_enu = tf.quaternion_apply(tf.quaternion_invert(initial_states[:,6:10].detach().clone()), accel_body)
#         print("Accel enu: ", accel_enu)
        omega = controls[:, 3:6] - initial_states[:,13:]
#         omega = torch.zeros(N, 3)

        pos_vel, pos_vel_cov = self.pos_vel_model(initial_states[:,:6], accel_enu)
        quat, quat_cov = self.quat_model(initial_states[:,6:10], omega)
        imu_bias, imu_bias_cov = self.imu_bias_model(initial_states[:,10:], None)

        # Combine the covariance matrices
        cholseky_matrix = torch.zeros((N, 16, 15))
        cholseky_matrix[:, 0:6, 0:6] = pos_vel_cov
        cholseky_matrix[:, 6:10, 6:9] = quat_cov
        cholseky_matrix[:, 10:, 9:] = imu_bias_cov

        return torch.cat((pos_vel, quat, imu_bias), axis=1), cholseky_matrix

# Combine previously defined models into a single dynamics model
class PosVelQuatAmbBiasModel(tfilter.base.DynamicsModel):
    def __init__(self, N_dim=1, dt = 0.0025):
        # Initialise the dynamics model, which is a subclass of the base class.
        # The state dimension is an input.
        super().__init__(state_dim=16+N_dim)
        
        # Initialise the standard deviations of the noise for each state dimension.
        self.pos_vel_quat_bias_model = PosVelQuatBiasModel(dt=dt)
        self.amb_model = AmbiguityModel(state_dim=N_dim, dt=dt)
        
    def update(self, dt=None, pos_x_std=None, pos_y_std=None, pos_z_std=None, vel_x_std=None, vel_y_std=None, vel_z_std=None, r_std=None, p_std=None, y_std=None, amb_std=None, cs_amb_std=None, acc_bias_std=None, gyr_bias_std=None):
        # Update the standard deviations of the noise for each state dimension.
        self.pos_vel_quat_bias_model.update(dt=dt, pos_x_std=pos_x_std, pos_y_std=pos_y_std, pos_z_std=pos_z_std, vel_x_std=vel_x_std, vel_y_std=vel_y_std, vel_z_std=vel_z_std, r_std=r_std, p_std=p_std, y_std=y_std, acc_bias_std=acc_bias_std, gyr_bias_std=gyr_bias_std)
        self.amb_model.update(dt=dt, amb_std=amb_std, cs_amb_std=cs_amb_std)
        
    def forward(self, initial_states, controls):
        # The dynamics model is a simple random walk model.
        # The controls are null.
        N, state_dim = initial_states.shape
        self.state_dim = state_dim

        pos_vel_quat_bias, pos_vel_quat_bias_cov = self.pos_vel_quat_bias_model(initial_states[:,:16], controls)
        amb, amb_cov = self.amb_model(initial_states[:,16:], None)

        # Combine the covariance matrices
        cholseky_matrix = torch.zeros((N, state_dim, state_dim-1))
        cholseky_matrix[:, 0:16, 0:15] = pos_vel_quat_bias_cov
        cholseky_matrix[:, 16:, 15:] = amb_cov

        return torch.cat((pos_vel_quat_bias, amb), axis=1), cholseky_matrix    
    
    
class CarFullPoseDynamicsModel(tfilter.base.DynamicsModel):
    def __init__(self, N_dim=31+32, dt = 0.0025):
        super().__init__(state_dim=10 + N_dim)
        self.N_dim = N_dim
        xy_std = torch.tensor(10.0)
        z_std = torch.tensor(0.005)
        q_std = torch.tensor([0.01, 0.01, 0.01, 0.01])
        v_std = torch.tensor([1.0, 1.0, 0.1])
        
        self.Qxy = torch.eye(2)*xy_std
        self.Qz = z_std
        self.Qq = torch.diag(q_std)
        self.Qv = torch.diag(v_std)
        
        if self.N_dim > 0:
            self.QN = torch.eye(self.N_dim)  # N
        
        self.update_dt_cov(dt, xy_std, z_std, q_std, v_std)
        
#         self.save_data = {
#             'a_enu': [],
#             'q_dot': []
#         }
    
    def update_ambiguity_cov(self, sigma):
        self.QN = torch.diag(sigma)
    
    def update_dt_cov(self, dt, xy_std, z_std, q_std, v_std):
        self.dt = dt
        self.Qxy = torch.eye(2)*xy_std
        self.Qz = z_std
        self.Qq = torch.diag(q_std)
        self.Qv = torch.diag(v_std)
        
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
        quat = torch.div(initial_states[:, 3:7], torch.norm(initial_states[:, 3:7], dim=1)[:, None]).detach()
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
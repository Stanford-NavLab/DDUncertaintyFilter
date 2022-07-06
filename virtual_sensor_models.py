import torchfilter as tfilter
import torch
import numpy as np

# (observation) -> (state, covariance)
class MyVirtualSensorModel(tfilter.base.VirtualSensorModel):
    def __init__(self, config):
        super().__init__(state_dim=self.state_dim, observation_dim = config.observation_dim)
        assert self.observation_dim >= self.state_dim
        self.H = config.H
            
        self.H_pinv = torch.pinverse(self.H) 
        self.R = config.R
        
    def forward(self, observations):
        N, observation_dim = observations.shape
        assert self.observation_dim == observation_dim
        predicted_state = (self.H_pinv[None, :, :] @ observations[:, :, None]).squeeze(-1)
        # Lower-triangular cholesky decomposition of covariance
        covariance = torch.cholesky(self.H_pinv @ self.R @ self.R.transpose(-1, -2) @ self.H_pinv.transpose(-1, -2))
        
        return predicted_state, covariance.expand((N, self.state_dim, self.state_dim))
    
    
class GNSSVirtualSensorModel(tfilter.base.VirtualSensorModel):
    def __init__(self, config, **kwargs):
        super().__init__(state_dim=config.state_dim)
        self.observation_dim = config.observation_dim
        self.pr_std = config.pr_std
        self.satXYZb = None
        self.idx_mask = np.zeros(self.state_dim, dtype=bool)
        self.idx_pos_mask = np.zeros(self.state_dim, dtype=bool)
        self.idx_b_mask = np.zeros(self.state_dim, dtype=bool)
        self.idx_pos_mask[config.idx.x] = True
        self.idx_pos_mask[config.idx.y] = True
        self.idx_pos_mask[config.idx.z] = True
        self.idx_b_mask[config.idx.b] = True
        
        assert self.observation_dim >= self.state_dim
        
        self.iterations = kwargs["iterations"]
        self.convergence = kwargs["convergence"]
        
    def update_sats(self, satXYZb):
        self.satXYZb = satXYZb
        self.observation_dim = len(satXYZb)
        
    def forward(self, observations):
        N, observation_dim = observations.shape
        assert self.observation_dim == observation_dim
        
        def expected_observations(x):
            return torch.linalg.norm(self.satXYZb[:, :3] - x[None, :3], dim=-1) + torch.abs(self.satXYZb[:, 3] - x[None, 3])
        
        predicted_state = torch.zeros(N, self.state_dim)
        covariance = torch.eye(self.state_dim).expand((N, self.state_dim, self.state_dim))
        
        for bidx, observation in enumerate(observations):
            initial = torch.zeros(4, requires_grad = True)
            for i in range(self.iterations): 
                previous_data = initial.clone()
                expec = expected_observations(initial)
                res = observation - expec
                jac = torch.autograd.functional.jacobian(expected_observations, initial.clone())
                # update 
                initial.data += (torch.linalg.pinv(jac) @ res[:, None]).squeeze(-1)
                # zero out current gradient to hold new gradients in next iteration 
                if torch.sum(torch.abs(initial - previous_data)) < self.convergence:
                    break
            predicted_state[bidx, self.idx_pos_mask] = initial[:3]
            predicted_state[bidx, self.idx_b_mask] = initial[3]
            # Lower-triangular cholesky decomposition of covariance
            R = torch.eye(self.observation_dim) * self.pr_std
            H = torch.autograd.functional.jacobian(expected_observations, predicted_state[bidx, :].clone())
            H_pinv = torch.linalg.pinv(H)
            covariance[bidx, :] = torch.cholesky(H_pinv @ R @ R.transpose(-1, -2) @ H_pinv.transpose(-1, -2) + torch.eye(self.state_dim)*1e-5)
        
        return predicted_state, covariance
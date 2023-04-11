import torchfilter as tfilter
from torchfilter import types
import torch
import numpy as np
from utils import *
from overrides import overrides
import fannypack as fp

class AsyncExtendedKalmanFilter(tfilter.filters.ExtendedKalmanFilter):
    """Differentiable EKF with asynchronous observation and controls forward pass.

    TODO: For building estimators with more complex observation spaces (eg images), see
    `VirtualSensorExtendedKalmanFilter`.
    """
    @overrides
    def forward(
        self,
        *,
        observations: types.ObservationsTorch,
        controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        """Kalman filter forward pass, single timestep.

        Args:
            observations (dict or torch.Tensor): Observation inputs. Should be either a
                dict of tensors or tensor of shape `(N, ...)`.
            controls (dict or torch.Tensor): Control inputs. Should be either a dict of
                tensors or tensor of shape `(N, ...)`.

        Returns:
            torch.Tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
        """
        # Check initialization
        assert self._initialized, "Kalman filter not initialized!"

        # Validate inputs
        N, state_dim = self.belief_mean.shape
        
        if controls is not None:
            assert fp.utils.SliceWrapper(controls).shape[0] == N
            # Predict step
            self._predict_step(controls=controls)

        if observations is not None:
            assert fp.utils.SliceWrapper(observations).shape[0] == N
            # Update step
            self._update_step(observations=observations)

        # Return mean
        return self.belief_mean
    
    def update_gnss(self, *args, **kwargs):
        self.measurement_model.update_gnss(*args, **kwargs)
        
    def update_imu(self, *args, **kwargs):
        self.measurement_model.update_imu(*args, **kwargs)
        
    def update_vo(self, *args, **kwargs):
        self.measurement_model.update_vo(*args, **kwargs)
        
    def update_vo_base(self, *args, **kwargs):
        self.measurement_model.update_vo_base(*args, **kwargs)
        
    def update_dynamics(self, *args, **kwargs):
        self.dynamics_model.update(*args, **kwargs)

    def get_covariance(self):
        return self.belief_covariance
    
    def get_state_statistics(self):
        return self.belief_mean, self.belief_covariance
    
class AsyncExtendedInformationFilter(tfilter.filters.ExtendedInformationFilter):
    """Differentiable EIF with asynchronous observation and controls forward pass.

    TODO: For building estimators with more complex observation spaces (eg images), see
    `VirtualSensorExtendedKalmanFilter`.
    """
    @overrides
    def forward(
        self,
        *,
        observations: types.ObservationsTorch,
        controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        """Kalman filter forward pass, single timestep.

        Args:
            observations (dict or torch.Tensor): Observation inputs. Should be either a
                dict of tensors or tensor of shape `(N, ...)`.
            controls (dict or torch.Tensor): Control inputs. Should be either a dict of
                tensors or tensor of shape `(N, ...)`.

        Returns:
            torch.Tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
        """
        # Check initialization
        assert self._initialized, "Kalman filter not initialized!"

        # Validate inputs
        N, state_dim = self.belief_mean.shape
        
        if controls is not None:
            assert fp.utils.SliceWrapper(controls).shape[0] == N
            # Predict step
            self._predict_step(controls=controls)

        if observations is not None:
            assert fp.utils.SliceWrapper(observations).shape[0] == N
            # Update step
            self._update_step(observations=observations)

        # Return mean
        return self.belief_mean
    
    
class AsyncParticleFilter(tfilter.filters.ParticleFilter):
    """Differentiable PF with asynchronous observation and controls forward pass."""
    
    @overrides
    def forward(
        self,
        *,
        observations: types.ObservationsTorch,
        controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        """Particle filter forward pass, single timestep.

        Args:
            observations (dict or torch.Tensor): observation inputs. should be
                either a dict of tensors or tensor of shape `(N, ...)`.
            controls (dict or torch.Tensor): control inputs. should be either a
                dict of tensors or tensor of shape `(N, ...)`.

        Returns:
            torch.Tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
        """

        # Make sure our particle filter's been initialized
        assert self._initialized, "Particle filter not initialized!"

        # Get our batch size (N), current particle count (M), & state dimension
        N, M, state_dim = self.particle_states.shape
        assert state_dim == self.state_dim

        # Decide whether or not we're resampling
        resample = self.resample
        if resample is None:
            # If not explicitly set, we disable resampling in train mode (to allow
            # gradients to propagate through time) and enable in eval mode (to prevent
            # particle deprivation)
            resample = not self.training

        # If we're not resampling and our current particle count doesn't match
        # our desired particle count, we need to either expand or contract our
        # particle set
        if not resample and self.num_particles != M:
            indices = self.particle_states.new_zeros(
                (N, self.num_particles), dtype=torch.long
            )

            # If output particles > our input particles, for the beginning part we copy
            # particles directly to reduce variance
            copy_count = (self.num_particles // M) * M
            if copy_count > 0:
                indices[:, :copy_count] = torch.arange(M).repeat(copy_count // M)[
                    None, :
                ]

            # For remaining particles, we sample w/o replacement (also lowers variance)
            remaining_count = self.num_particles - copy_count
            assert remaining_count >= 0
            if remaining_count > 0:
                indices[:, copy_count:] = torch.randperm(M, device=indices.device)[
                    None, :remaining_count
                ]

            # Gather new particles, weights
            M = self.num_particles
            self.particle_states = self.particle_states.gather(
                1, indices[:, :, None].expand((N, M, state_dim))
            )
            self.particle_log_weights = self.particle_log_weights.gather(1, indices)
            assert self.particle_states.shape == (N, self.num_particles, state_dim)
            assert self.particle_log_weights.shape == (N, self.num_particles)

            # Normalize particle weights to sum to 1.0
            self.particle_log_weights = self.particle_log_weights - torch.logsumexp(
                self.particle_log_weights, dim=1, keepdim=True
            )

        if controls is not None:
            assert len(fp.utils.SliceWrapper(controls)) == N
            # Propagate particles through our dynamics model
            # A bit of extra effort is required for the extra particle dimension
            # > For our states, we flatten along the N/M axes
            # > For our controls, we repeat each one `M` times, if M=3:
            #       [u0 u1 u2] should become [u0 u0 u0 u1 u1 u1 u2 u2 u2]
            #
            # Currently each of the M particles within a "sample" get the same action, but
            # we could also add noise in the action space (a la Jonschkowski et al. 2018)
            reshaped_states = self.particle_states.reshape(-1, self.state_dim)
            reshaped_controls = fp.utils.SliceWrapper(controls).map(
                lambda tensor: torch.repeat_interleave(tensor, repeats=M, dim=0)
            )
            predicted_states, scale_trils = self.dynamics_model(
                initial_states=reshaped_states, controls=reshaped_controls
            )
            self.particle_states = (
                torch.distributions.MultivariateNormal(
                    loc=predicted_states, scale_tril=scale_trils
                )
                .rsample()  # Note that we use `rsample` to make sampling differentiable
                .view(N, M, self.state_dim)
            )
            assert self.particle_states.shape == (N, M, self.state_dim)

        if observations is not None:
            # Re-weight particles using observations
            self.particle_log_weights = self.particle_log_weights + self.measurement_model(
                states=self.particle_states,
                observations=observations,
            )

            # Normalize particle weights to sum to 1.0
            self.particle_log_weights = self.particle_log_weights - torch.logsumexp(
                self.particle_log_weights, dim=1, keepdim=True
            )

        # Compute output
        state_estimates: types.StatesTorch
        if self.estimation_method == "weighted_average":
            state_estimates = torch.sum(
                torch.exp(self.particle_log_weights[:, :, np.newaxis])
                * self.particle_states,
                dim=1,
            )
        elif self.estimation_method == "argmax":
            best_indices = torch.argmax(self.particle_log_weights, dim=1)
            state_estimates = torch.gather(
                self.particle_states, dim=1, index=best_indices
            )
        else:
            assert False, "Unsupported estimation method!"

        # Resampling
        if resample:
            self._resample()

        # Post-condition :)
        assert state_estimates.shape == (N, state_dim)
        assert self.particle_states.shape == (N, self.num_particles, state_dim)
        assert self.particle_log_weights.shape == (N, self.num_particles)

        return state_estimates

##########################################################################################################################
# RBPF
##########################################################################################################################
    
class AsyncRaoBlackwellizedParticleFilter(tfilter.filters.ParticleFilter):
    """Differentiable RBPF with asynchronous observation and controls forward pass."""
    
    def attach_ekf(self, filter_base, pf_state_mask, mode='linearization_points'):
        self.ekf = filter_base
        self.pf_state_mask = pf_state_mask
        self.pf_state_dim = torch.sum(pf_state_mask)
        self.mode = mode        # ['redundant', 'bank', 'linearization_points', 'naive']
        
    @overrides
    def initialize_beliefs(
        self, *, mean: types.StatesTorch, covariance: types.CovarianceTorch
    ) -> None:
        """Populates initial particles, which will be normally distributed.

        Args:
            mean (torch.Tensor): Mean of belief. Shape should be
                `(N, state_dim)`.
            covariance (torch.Tensor): Covariance of belief. Shape should be
                `(N, state_dim, state_dim)`.
        """
        N = mean.shape[0]
        assert mean.shape == (N, self.state_dim)
        assert covariance.shape == (N, self.state_dim, self.state_dim)
        M = self.num_particles

        # Particle filter uncertainty parameters
        self.pf_state_uncertainty = 1 + 0e-1
        self.resample_factor = 0.5
        self.jitter_dynamics = 1 + 0e-1
        
        # Sample particles (assumes pf_state_mask has initial states and kf has later ones)
        pf_covariance = covariance[:, :self.pf_state_dim, :self.pf_state_dim]
        
        particle_states = (
            torch.distributions.MultivariateNormal(mean[:, :self.pf_state_dim], pf_covariance)
            .sample((M,))
            .transpose(0, 1)
        )
        
        self.particle_states = torch.cat((particle_states, mean.expand(N, M, self.state_dim)[:, :, self.pf_state_dim:]), -1)
        assert self.particle_states.shape == (N, M, self.state_dim)
        
        # Initialize EKF with generated particles
        kf_covariance = covariance[:, None, :, :].expand(N, M, self.state_dim, self.state_dim)
        
        ekf_states = self.particle_states.reshape(-1, self.state_dim)
        ekf_covariance = kf_covariance.reshape(-1, self.state_dim, self.state_dim)
        self.ekf.initialize_beliefs(mean=ekf_states, covariance=ekf_covariance)

        # Normalize weights
        self.particle_log_weights = self.particle_states.new_full(
            (N, M), float(-np.log(M, dtype=np.float32))
        )
        assert self.particle_log_weights.shape == (N, M)

        # Set initialized flag
        self._initialized = True
        
        # EKF covariance reset
        self.kf_covariance = kf_covariance.detach().clone() # (N, M, state_dim, state_dim)
    
    @overrides
    def forward(
        self,
        *,
        observations: types.ObservationsTorch,
        controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        """RB Particle filter forward pass, single timestep.

        Args:
            observations (dict or torch.Tensor): observation inputs. should be
                either a dict of tensors or tensor of shape `(N, ...)`.
            controls (dict or torch.Tensor): control inputs. should be either a
                dict of tensors or tensor of shape `(N, ...)`.

        Returns:
            torch.Tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
        """

        # Make sure our filter's been initialized
        assert self._initialized, "Filter not initialized!"

        # Get our batch size (N), current particle count (M), & state dimension
        N, M, state_dim = self.particle_states.shape
        assert state_dim == self.state_dim

        # Decide whether or not we're resampling
        resample = self.resample and not self.mode=='bank'
        if resample is None:
            # If not explicitly set, we disable resampling in train mode (to allow
            # gradients to propagate through time) and enable in eval mode (to prevent
            # particle deprivation)
            resample = not self.training

        # If we're not resampling and our current particle count doesn't match
        # our desired particle count, we need to either expand or contract our
        # particle set
        if not resample and self.num_particles != M:
            indices = self.particle_states.new_zeros(
                (N, self.num_particles), dtype=torch.long
            )

            # If output particles > our input particles, for the beginning part we copy
            # particles directly to reduce variance
            copy_count = (self.num_particles // M) * M
            if copy_count > 0:
                indices[:, :copy_count] = torch.arange(M).repeat(copy_count // M)[
                    None, :
                ]

            # For remaining particles, we sample w/o replacement (also lowers variance)
            remaining_count = self.num_particles - copy_count
            assert remaining_count >= 0
            if remaining_count > 0:
                indices[:, copy_count:] = torch.randperm(M, device=indices.device)[
                    None, :remaining_count
                ]

            # Gather new particles, weights
            M = self.num_particles
            self.particle_states = self.particle_states.gather(
                1, indices[:, :, None].expand((N, M, state_dim))
            )
            self.particle_log_weights = self.particle_log_weights.gather(1, indices)
            self.kf_covariance = self.ekf.get_covariance().reshape(N, M, state_dim, state_dim).gather(
                1, indices[:, :, None, None].expand((N, M, state_dim, state_dim))
            )
            
            assert self.particle_states.shape == (N, self.num_particles, state_dim)
            assert self.particle_log_weights.shape == (N, self.num_particles)

            # Normalize particle weights to sum to 1.0
            self.particle_log_weights = self.particle_log_weights - torch.logsumexp(
                self.particle_log_weights, dim=1, keepdim=True
            )
        
        if controls is not None:
            assert len(fp.utils.SliceWrapper(controls)) == N
            # Propagate particles through our dynamics model
            # A bit of extra effort is required for the extra particle dimension
            # > For our states, we flatten along the N/M axes
            # > For our controls, we repeat each one `M` times, if M=3:
            #       [u0 u1 u2] should become [u0 u0 u0 u1 u1 u1 u2 u2 u2]
            #
            # Currently each of the M particles within a "sample" get the same action, but
            # we could also add noise in the action space (a la Jonschkowski et al. 2018)
            reshaped_states = self.particle_states.reshape(-1, self.state_dim)
            reshaped_covariance = self.kf_covariance.reshape(-1, self.state_dim, self.state_dim)
            reshaped_controls = fp.utils.SliceWrapper(controls).map(
                lambda tensor: torch.repeat_interleave(tensor, repeats=M, dim=0)
            )
            self.ekf._belief_mean = reshaped_states
            
            self.ekf._belief_covariance = reshaped_covariance
                        
            predicted_states, process_Q = self.dynamics_model(reshaped_states, reshaped_controls)

            process_Q_pf = self.jitter_dynamics*process_Q[:, :self.pf_state_dim, :self.pf_state_dim]
            
            # Propagate PF state parts
            if self.mode=='bank':
                self.particle_states = predicted_states[:, :self.pf_state_dim]
            elif self.mode=='linearization_points':
                self.particle_states = (
                    torch.distributions.MultivariateNormal(
                        loc=predicted_states[:, :self.pf_state_dim], scale_tril=process_Q_pf
                    )
                    .rsample()  # Note that we use `rsample` to make sampling differentiable
                )
                # self.particle_states = predicted_states[:, :self.pf_state_dim]
            
            # Run EKF predict step
            predicted_states = self.ekf(controls=reshaped_controls, observations=None)

            # Concatenate with EKF state parts
            self.particle_states = torch.cat((self.particle_states, predicted_states[:, self.pf_state_dim:]), -1).view(N, M, self.state_dim)
            self.kf_covariance = self.ekf.get_covariance().reshape(N, M, state_dim, state_dim)
            
            assert self.particle_states.shape == (N, M, self.state_dim)
            assert self.kf_covariance.shape == (N, M, self.state_dim, self.state_dim)

        meas_log_wts = None
            
        if observations is not None:
            reshaped_states = self.particle_states.reshape(-1, self.state_dim)
            reshaped_covariance = self.kf_covariance.reshape(-1, self.state_dim, self.state_dim)
            reshaped_observations = fp.utils.SliceWrapper(observations).map(
                lambda tensor: torch.repeat_interleave(tensor, repeats=M, dim=0)
            )
            self.ekf._belief_mean = reshaped_states
            self.ekf._belief_covariance = reshaped_covariance
            
            # print(self.ekf.measurement_model.observation_dim)
            corrected_states = self.ekf(controls=None, observations=reshaped_observations)
      
            # Retain particle states or use full EKF states
#             self.particle_states = torch.cat((reshaped_states[:, self.pf_state_mask], corrected_states[:, ~self.pf_state_mask]), -1).reshape(N, M, self.state_dim)
            self.particle_states = corrected_states.reshape(N, M, self.state_dim)
            self.kf_covariance = self.ekf.get_covariance().reshape(N, M, state_dim, state_dim)
            
            # Re-weight particles using observations (1-point quadrature)
            meas_log_wts = self.measurement_model(
                    states=self.particle_states,
                    observations=observations,
            )
                
            self.particle_log_weights = self.particle_log_weights + meas_log_wts

#                 # Sample EKF states for integration
#                 N_samples = 10
                
#                 ekf_state_samples = (
#                     torch.distributions.MultivariateNormal(
#                         loc=corrected_states, scale_tril=torch.tril(self.ekf._belief_covariance)
#                     )
#                     .sample(sample_shape=(N_samples, ))  # Note that we use `rsample` to make sampling differentiable
#                     .transpose(0, 1).transpose(1, 2)
#                 )

#                 ekf_state_samples = ekf_state_samples.view(N, M*N_samples, self.state_dim)

#                 # Re-weight particles using observations (MC integration across EKF states)
#                 self.particle_log_weights = self.particle_log_weights + torch.logsumexp(self.measurement_model(
#                     states=ekf_state_samples,
#                     observations=observations,
#                 ).view(N, M, 10), dim=-1)*(1/N_samples)
                
            # Normalize particle weights to sum to 1.0
            self.particle_log_weights = self.particle_log_weights - torch.logsumexp(
                self.particle_log_weights, dim=1, keepdim=True
            )

        # Compute output
        state_estimates: types.StatesTorch
        if self.estimation_method == "weighted_average":
            state_estimates = torch.sum(
                torch.exp(self.particle_log_weights[:, :, np.newaxis])
                * self.particle_states,
                dim=1,
            )
        elif self.estimation_method == "argmax":
            best_indices = torch.argmax(self.particle_log_weights, dim=1)
            state_estimates = torch.gather(
                self.particle_states, dim=1, index=best_indices.expand(N, 1, state_dim)
            ).reshape(N, state_dim)
        else:
            assert False, "Unsupported estimation method!"

        # Resampling
        if resample and self.effective_log_sample_size() < np.log(N) + np.log(M) + np.log(self.resample_factor):
            self._resample()

        # Post-condition :)
        assert state_estimates.shape == (N, state_dim)
        assert self.particle_states.shape == (N, self.num_particles, state_dim)
        assert self.particle_log_weights.shape == (N, self.num_particles)

        return state_estimates
    
    def _resample(self) -> None:
        """Resample particles."""
        # Note the distinction between `M`, the current number of particles, and
        # `self.num_particles`, the desired number of particles
        N, M, state_dim = self.particle_states.shape

        sample_logits: torch.Tensor
        uniform_log_weights = self.particle_log_weights.new_full(
            (N, self.num_particles), float(-np.log(M, dtype=np.float32))
        )
        if self.soft_resample_alpha < 1.0:
            # Soft resampling
            assert self.particle_log_weights.shape == (N, M)
            sample_logits = torch.logsumexp(
                torch.stack(
                    [
                        self.particle_log_weights + np.log(self.soft_resample_alpha),
                        uniform_log_weights + np.log(1.0 - self.soft_resample_alpha),
                    ],
                    dim=0,
                ),
                dim=0,
            )
            self.particle_log_weights = self.particle_log_weights - sample_logits
        else:
            # Standard particle filter re-sampling -- this stops gradients
            # This is the most naive flavor of resampling, and not the low
            # variance approach
            #
            # Note the distinction between M, the current # of particles,
            # and self.num_particles, the desired # of particles
            sample_logits = self.particle_log_weights
            self.particle_log_weights = uniform_log_weights

        assert sample_logits.shape == (N, M)
        distribution = torch.distributions.Categorical(logits=sample_logits)
        state_indices = distribution.sample((self.num_particles,)).T
        assert state_indices.shape == (N, self.num_particles)

        self.particle_states = torch.gather(
            self.particle_states,
            dim=1,
            index=state_indices[:, :, None].expand((N, self.num_particles, state_dim)),
        )
        self.kf_covariance = self.ekf.get_covariance().reshape(N, self.num_particles, state_dim, state_dim).gather(
                1, state_indices[:, :, None, None].expand((N, self.num_particles, state_dim, state_dim))
            )
        # # ^This gather magic is equivalent to:
        # particle_states_alt = torch.zeros_like(self.particle_states)
        # for i in range(N):
        #     particle_states_alt[i] = self.particle_states[i][state_indices[i]]

    def effective_log_sample_size(self):
        return -torch.logsumexp(2*self.particle_log_weights, dim=(0, 1), keepdim=False)
    
    def update_gnss(self, *args, **kwargs):
        self.measurement_model.update_gnss(*args, **kwargs)
        self.ekf.measurement_model.update_gnss(*args, **kwargs)
        
    def update_imu(self, *args, **kwargs):
        self.measurement_model.update_imu(*args, **kwargs)
        self.ekf.measurement_model.update_imu(*args, **kwargs)
        
    def update_vo(self, *args, **kwargs):
        self.measurement_model.update_vo(*args, **kwargs)
        self.ekf.measurement_model.update_vo(*args, **kwargs)
        
    def update_vo_base(self, *args, **kwargs):
        self.measurement_model.update_vo_base(*args, **kwargs)
        self.ekf.measurement_model.update_vo_base(*args, **kwargs)
        
    def update_dynamics(self, *args, **kwargs):
        self.dynamics_model.update(*args, **kwargs)
        self.ekf.dynamics_model.update(*args, **kwargs)

    def get_covariance(self):
        # convert self.particle_log_weights to weights
        weights = torch.exp(self.particle_log_weights)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        weights = weights.expand(-1, -1, self.state_dim, self.state_dim)
        # compute weighted covariance
        covariance = torch.sum(weights * self.kf_covariance, dim=1, keepdim=False)
        return covariance
    
    def get_empirical_covariance(self):
        # convert self.particle_log_weights to weights
        weights = torch.exp(self.particle_log_weights)
        weights = weights.unsqueeze(-1)
        estimated_state = torch.sum(
                weights
                * self.particle_states,
                dim=1,
            )
        state_diff = self.particle_states - estimated_state.unsqueeze(1)
        # compute weighted empirical covariance
        covariance = torch.sum(weights * state_diff * state_diff.transpose(1, 2), dim=1, keepdim=False)
        return covariance
    
    def get_state_statistics(self):
        return self.particle_states[0, :, :], self.kf_covariance[0, :, :, :]
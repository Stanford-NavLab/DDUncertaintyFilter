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
    
    def attach_ekf(self, dynamics_model, measurement_model: tfilter.base.KalmanFilterMeasurementModel, pf_state_mask, bank_mode=False):
        self.ekf = AsyncExtendedKalmanFilter(
                dynamics_model=dynamics_model,
                measurement_model=measurement_model,
                )
        self.pf_state_mask = pf_state_mask
        self.pf_state_dim = torch.sum(pf_state_mask)
        self.bank_mode = bank_mode
    
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

        # Particle filter uncertainty parameter
        self.pf_state_uncertainty = 1 - 1e-2
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
        kf_covariance = self.pf_state_uncertainty*torch.eye(self.state_dim).expand(N*M, self.state_dim, self.state_dim)
        kf_covariance[:, self.pf_state_dim:, self.pf_state_dim:] =  covariance[:, self.pf_state_dim:, self.pf_state_dim:]
        
        ekf_states = self.particle_states.reshape(-1, self.state_dim)
        self.ekf.initialize_beliefs(mean=ekf_states, covariance=kf_covariance)

        # Normalize weights
        self.particle_log_weights = self.particle_states.new_full(
            (N, M), float(-np.log(M, dtype=np.float32))
        )
        assert self.particle_log_weights.shape == (N, M)

        # Set initialized flag
        self._initialized = True
        
        # EKF covariance reset
        self.kf_covariance = self.ekf._belief_covariance.detach()
        self.init_covariance = kf_covariance.detach()
    
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

        # Make sure our particle filter's been initialized
        assert self._initialized, "Particle filter not initialized!"

        self.tmp_outs = {}
        
        # Get our batch size (N), current particle count (M), & state dimension
        N, M, state_dim = self.particle_states.shape
        assert state_dim == self.state_dim

        # Decide whether or not we're resampling
        resample = self.resample and not self.bank_mode
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
            self.kf_covariance = self.kf_covariance.gather(1, indices.reshape(-1)[:, None, None].expand((N*M, state_dim, state_dim)))
            
            assert self.particle_states.shape == (N, self.num_particles, state_dim)
            assert self.particle_log_weights.shape == (N, self.num_particles)

            # Normalize particle weights to sum to 1.0
            self.particle_log_weights = self.particle_log_weights - torch.logsumexp(
                self.particle_log_weights, dim=1, keepdim=True
            )
        self.tmp_outs['initial states'] = self.particle_states.detach()
        self.tmp_outs['initial logwt'] = self.particle_log_weights.detach()
        self.tmp_outs['initial cov'] = self.ekf._belief_covariance.detach()
            
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
            self.ekf._belief_mean = reshaped_states
            
            if not self.bank_mode:
                self.ekf._belief_covariance = self.kf_covariance.detach()
            
            predicted_states = self.ekf(controls=reshaped_controls, observations=None)
            
            scale_trils = self.jitter_dynamics*self.dynamics_model.calc_Q(dim=self.pf_state_dim).expand((N*M, self.pf_state_dim, self.pf_state_dim))
            
            # Propagate PF state parts
            if self.bank_mode:
                self.particle_states = predicted_states[:, :self.pf_state_dim]
            else:
                self.particle_states = (
                    torch.distributions.MultivariateNormal(
                        loc=predicted_states[:, :self.pf_state_dim], scale_tril=scale_trils
                    )
                    .rsample()  # Note that we use `rsample` to make sampling differentiable
                )
            
            # Concatenate with EKF state parts
            self.particle_states = torch.cat((self.particle_states, predicted_states[:, self.pf_state_dim:]), -1).view(N, M, self.state_dim)
            self.kf_covariance = self.ekf._belief_covariance.detach()
            self.kf_covariance[:, :self.pf_state_dim, :self.pf_state_dim] *= self.pf_state_uncertainty
            
            assert self.particle_states.shape == (N, M, self.state_dim)

            self.tmp_outs['predicted state'] = self.particle_states
            self.tmp_outs['predicted logwt'] = self.particle_log_weights
            self.tmp_outs['predicted cov'] = self.ekf._belief_covariance.detach()
        meas_log_wts = None
            
        if observations is not None:
            reshaped_states = self.particle_states.reshape(-1, self.state_dim)
            
            reshaped_observations = fp.utils.SliceWrapper(observations).map(
                lambda tensor: torch.repeat_interleave(tensor, repeats=M, dim=0)
            )
            
            self.ekf._belief_mean = reshaped_states
            
            self.ekf._belief_covariance = self.kf_covariance.detach()
            
            corrected_states = self.ekf(controls=None, observations=reshaped_observations)
      
            # Retain particle states or use full EKF states
#             self.particle_states = torch.cat((reshaped_states[:, self.pf_state_mask], corrected_states[:, ~self.pf_state_mask]), -1).reshape(N, M, self.state_dim)
            self.particle_states = corrected_states.reshape(N, M, self.state_dim)
            self.kf_covariance = self.ekf._belief_covariance.detach()
            
            if self.measurement_model.kalman_filter_measurement_model.mode == "gnss":
                # Re-weight particles using observations (1-point quadrature)
                meas_log_wts = self.measurement_model(
                    states=self.particle_states,
                    observations=observations,
                )
                
                self.particle_log_weights = self.particle_log_weights + meas_log_wts + np.log(2)

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


                self.tmp_outs['corrected state'] = self.particle_states
                self.tmp_outs['corrected logwt'] = self.particle_log_weights
                self.tmp_outs['corrected cov'] = self.ekf._belief_covariance.detach()
        
        # Compute output
        state_estimates: types.StatesTorch
        if self.estimation_method == "weighted_average":
            state_estimates = torch.sum(
                torch.exp(self.particle_log_weights[:, :, np.newaxis])
                * self.particle_states,
                dim=1,
            )
#             robust_weights = torch.exp(self.particle_log_weights).detach().clone()
#             robust_weights[robust_weights<float(-np.log(M, dtype=np.float32))] = -1e10
#             robust_weights -= torch.logsumexp(
#                     robust_weights, dim=1, keepdim=True
#                 )
#             state_estimates = torch.sum(
#                 robust_weights[:, :, np.newaxis]
#                 * self.particle_states,
#                 dim=1,
#             )
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
            self.kf_covariance = self.init_covariance.detach()
            
        # Post-condition :)
        assert state_estimates.shape == (N, state_dim)
        assert self.particle_states.shape == (N, self.num_particles, state_dim)
        assert self.particle_log_weights.shape == (N, self.num_particles)

        return state_estimates
    
    def effective_log_sample_size(self):
        return -torch.logsumexp(2*self.particle_log_weights, dim=(0, 1), keepdim=False)
    
    def update_sats(self, *args, pf_std=(None, None, None), kf_std=(None, None, None), **kwargs):
        self.measurement_model.update_sats(*args, prange_std=pf_std[0], carrier_noN_std=pf_std[1], carrier_N_std=pf_std[2], **kwargs)
        self.ekf.measurement_model.update_sats(*args, prange_std=kf_std[0], carrier_noN_std=kf_std[1], carrier_N_std=kf_std[2], **kwargs)
    
    def update_imu_std(self, *args, **kwargs):
        self.measurement_model.update_imu_std(*args, **kwargs)
        self.ekf.measurement_model.update_imu_std(*args, **kwargs)
        
    def update_vo_std(self, *args, **kwargs):
        self.measurement_model.update_vo_std(*args, **kwargs)
        self.ekf.measurement_model.update_vo_std(*args, **kwargs)
        
    def update_dt_cov(self, *args, **kwargs):
        self.dynamics_model.update_dt_cov(*args, **kwargs)
        self.ekf.dynamics_model.update_dt_cov(*args, **kwargs)
    
    def update_ambiguity_cov(self, *args, **kwargs):
        self.dynamics_model.update_ambiguity_cov(*args, **kwargs)
        self.ekf.dynamics_model.update_ambiguity_cov(*args, **kwargs)
import torchfilter as tfilter
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
        observations: tfilter.types.ObservationsTorch,
        controls: tfilter.types.ControlsTorch,
    ) -> tfilter.types.StatesTorch:
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
    
import torchfilter as tfilter
import torch
import numpy as np
from kf_measurement_models import *

# (particles, observation) -> (log-likelihoods)
class MyPFMeasurementModelA(tfilter.base.ParticleFilterMeasurementModelWrapper):
    def __init__(self, config):
        super().__init__(
            kalman_filter_measurement_model=MyKFMeasurementModel(config)
        )
        
class GNSSPFMeasurementModelA(tfilter.base.ParticleFilterMeasurementModelWrapper):
    def __init__(self, config):
        super().__init__(
            kalman_filter_measurement_model=GNSSKFMeasurementModel(config)
        )
        
class GNSSPFMeasurementModelDD(tfilter.base.ParticleFilterMeasurementModelWrapper):
    def __init__(self, base_pos, N_dim=0):
        super().__init__(
            kalman_filter_measurement_model=GNSSDDKFMeasurementModel(base_pos, N_dim=N_dim)
        )
        self.update_sats = self.kalman_filter_measurement_model.update_sats
        
class GNSSPFMeasurementModel_IMU_DD(tfilter.base.ParticleFilterMeasurementModelWrapper):
    def __init__(self, base_pos, N_dim=0):
        super().__init__(
            kalman_filter_measurement_model=IMUDDMeasurementModel(base_pos, N_dim=N_dim)
        )
        self.update_sats = self.kalman_filter_measurement_model.update_sats
        self.update_imu_std = self.kalman_filter_measurement_model.update_imu_std
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
    def __init__(self, *args, **kwargs):
        super().__init__(
            kalman_filter_measurement_model=GNSSDDKFMeasurementModel(*args, **kwargs)
        )
        self.update_sats = self.kalman_filter_measurement_model.update_sats
        
class GNSSPFMeasurementModel_IMU_DD(tfilter.base.ParticleFilterMeasurementModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(
            kalman_filter_measurement_model=IMUDDMeasurementModel(*args, **kwargs)
        )
        self.update_sats = self.kalman_filter_measurement_model.update_sats
        self.update_imu_std = self.kalman_filter_measurement_model.update_imu_std
        
class GNSSPFMeasurementModel_IMU_DD_VO(tfilter.base.ParticleFilterMeasurementModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(
            kalman_filter_measurement_model=IMU_VO_DD_MeasurementModel(*args, **kwargs)
        )
        self.update_sats = self.kalman_filter_measurement_model.update_sats
        self.update_imu_std = self.kalman_filter_measurement_model.update_imu_std
        self.update_vo_std = self.kalman_filter_measurement_model.update_vo_std
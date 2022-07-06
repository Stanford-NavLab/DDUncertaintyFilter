import torchfilter as tfilter
import torch
import numpy as np
from kf_measurement_models import MyKFMeasurementModel, GNSSKFMeasurementModel

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
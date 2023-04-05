import torchfilter as tfilter
import torch
import numpy as np
import pytorch3d.transforms as tf
import torch.autograd.functional as F
from utils import *

class MEstimationModel(object):
    def __init__(self, threshold=0.8, debug=False):
        self.threshold = threshold
        self.debug = debug
    
    def generate_outlier_mask(self, residuals):
        if self.debug:
            print("residuals", residuals)
        return torch.abs(residuals) > self.threshold
import pandas as pd
import numpy as np
from threading import Thread
import cv2
import os, sys
import time
import glob
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
from itertools import chain, compress
from queue import Queue

# MSCKF config
class OptimizationConfigKITTI(object):
    """
    Configuration parameters for 3d feature position optimization.
    """
    def __init__(self):
        self.translation_threshold = -1.0  # 0.2
        self.huber_epsilon = 0.01
        self.estimation_precision = 5e-7
        self.initial_damping = 1e-3
        self.outer_loop_max_iteration = 5 # 10
        self.inner_loop_max_iteration = 5 # 10

# All configuration parameters
class ConfigKITTI(object):
    def __init__(self):
        # feature position optimization
        self.optimization_config = OptimizationConfigKITTI()

        ## image processor
        self.grid_row = 4
        self.grid_col = 5
        self.grid_num = self.grid_row * self.grid_col
        self.grid_min_feature_num = 3
        self.grid_max_feature_num = 5
        self.fast_threshold = 15
        self.ransac_threshold = 3
        self.stereo_threshold = 5
        self.max_iteration = 30
        self.track_precision = 0.01
        self.pyramid_levels = 3
        self.patch_size = 15
        self.win_size = (self.patch_size, self.patch_size)

        self.lk_params = dict(
            winSize=self.win_size,
            maxLevel=self.pyramid_levels,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                self.max_iteration, 
                self.track_precision),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        

        ## msckf vio
        # gravity
        self.gravity_acc = 9.81
        self.gravity = np.array([0.0, 0.0, -self.gravity_acc])

        # Framte rate of the stereo images. This variable is only used to 
        # determine the timing threshold of each iteration of the filter.
        self.frame_rate = 20

        # Maximum number of camera states to be stored
        self.max_cam_state_size = 20

        # The position uncertainty threshold is used to determine
        # when to reset the system online. Otherwise, the ever-increaseing
        # uncertainty will make the estimation unstable.
        # Note this online reset will be some dead-reckoning.
        # Set this threshold to nonpositive to disable online reset.
        self.position_std_threshold = 8.0

        # Threshold for determine keyframes
        self.rotation_threshold = 0.2618
        self.translation_threshold = 0.4
        self.tracking_rate_threshold = 0.5

        # Noise related parameters (Use variance instead of standard deviation)
        self.gyro_noise = 10.0 ** 2
        self.acc_noise = 10.0 ** 2
        self.gyro_bias_noise = 1e-5 ** 2
        self.acc_bias_noise = 1e-5 ** 2
        self.observation_noise = 0.035 ** 2

        # initial state
        self.velocity = np.zeros(3)

        # The initial covariance of orientation and position can be
        # set to 0. But for velocity, bias and extrinsic parameters, 
        # there should be nontrivial uncertainty.
        self.velocity_cov = 0.25
        self.gyro_bias_cov = 1e-5
        self.acc_bias_cov = 1e-5
        self.extrinsic_rotation_cov = 3.0462e-4
        self.extrinsic_translation_cov = 2.5e-5

        ## calibration parameters
        # T_imu_cam: takes a vector from the IMU frame to the cam frame.
        # see https://github.com/ethz-asl/kalibr/wiki/yaml-formats
        # TODO: update to correct KITTI transform
        
        
        # self.T_imu_cam0 = np.array([
        #     [ 0.014865542981794,   0.999557249008346,  -0.025774436697440,   0.065222909535531],
        #     [-0.999880929698575,   0.014967213324719,   0.003756188357967,  -0.020706385492719],
        #     [ 0.004140296794224,   0.025715529947966,   0.999660727177902,  -0.008054602460030],
        #     [                 0,                   0,                   0,   1.000000000000000]])
        self.T_imu_cam0 = np.identity(4)
        self.cam0_camera_model = 'pinhole'
        self.cam0_distortion_model = 'radtan'
        self.cam0_distortion_coeffs = np.array(
            [0.0, 0.0, 0.0, 0.0])
        self.cam0_intrinsics = np.array([718.856, 718.856, 607.1928, 185.2157])
        self.cam0_resolution = np.array([1241, 376])

        self.T_imu_body = np.identity(4)
        
class ConfigKITTI360(object):
    def __init__(self):
        # feature position optimization
        self.optimization_config = OptimizationConfigKITTI()

        ## image processor
        self.grid_row = 4
        self.grid_col = 5
        self.grid_num = self.grid_row * self.grid_col
        self.grid_min_feature_num = 3
        self.grid_max_feature_num = 5
        self.fast_threshold = 15
        self.ransac_threshold = 3
        self.stereo_threshold = 5
        self.max_iteration = 30
        self.track_precision = 0.01
        self.pyramid_levels = 3
        self.patch_size = 15
        self.win_size = (self.patch_size, self.patch_size)

        self.lk_params = dict(
            winSize=self.win_size,
            maxLevel=self.pyramid_levels,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                self.max_iteration, 
                self.track_precision),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        

        ## msckf vio
        # gravity
        self.gravity_acc = 9.81
        self.gravity = np.array([0.0, 0.0, -self.gravity_acc])

        # Framte rate of the stereo images. This variable is only used to 
        # determine the timing threshold of each iteration of the filter.
        self.frame_rate = 20

        # Maximum number of camera states to be stored
        self.max_cam_state_size = 20

        # The position uncertainty threshold is used to determine
        # when to reset the system online. Otherwise, the ever-increaseing
        # uncertainty will make the estimation unstable.
        # Note this online reset will be some dead-reckoning.
        # Set this threshold to nonpositive to disable online reset.
        self.position_std_threshold = 8.0

        # Threshold for determine keyframes
        self.rotation_threshold = 0.2618
        self.translation_threshold = 0.4
        self.tracking_rate_threshold = 0.5

        # Noise related parameters (Use variance instead of standard deviation)
        self.gyro_noise = 10.0 ** 2
        self.acc_noise = 10.0 ** 2
        self.gyro_bias_noise = 1e-5 ** 2
        self.acc_bias_noise = 1e-5 ** 2
        self.observation_noise = 0.035 ** 2

        # initial state
        self.velocity = np.zeros(3)

        # The initial covariance of orientation and position can be
        # set to 0. But for velocity, bias and extrinsic parameters, 
        # there should be nontrivial uncertainty.
        self.velocity_cov = 0.25
        self.gyro_bias_cov = 1e-5
        self.acc_bias_cov = 1e-5
        self.extrinsic_rotation_cov = 3.0462e-4
        self.extrinsic_translation_cov = 2.5e-5

        ## calibration parameters
        # T_imu_cam: takes a vector from the IMU frame to the cam frame.
        # see https://github.com/ethz-asl/kalibr/wiki/yaml-formats
        # TODO: update to correct KITTI transform
        
        
        self.T_cam0_imu = np.array([
            [ 0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])
        self.T_imu_cam0 = np.linalg.inv(self.T_cam0_imu)
        self.cam0_camera_model = 'pinhole'
        self.cam0_distortion_model = 'radtan'
        self.cam0_distortion_coeffs = np.array(
            [0.0, 0.0, 0.0, 0.0])
        self.cam0_intrinsics = np.array([552.554261, 552.554261, 682.049453, 238.769549])
        self.cam0_resolution = np.array([1392, 512])

        self.T_imu_body = np.identity(4)
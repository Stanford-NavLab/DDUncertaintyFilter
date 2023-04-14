import torchfilter as tfilter
import torch.autograd.functional as F
import numpy as np
import pandas as pd
import pymap3d as pm
from matplotlib import pyplot as plt
import plotly.express as px
import pickle
import torch
import torch.optim as optim
from tqdm import tqdm
from collections import namedtuple
import pytorch3d.transforms as tf
import xarray as xr
from torch.distributions.multivariate_normal import MultivariateNormal

import os, sys
sys.path.append('/scratch/users/shubhgup/1_18_winter/DDUncertaintyFilter/')
# sys.path.append('..')

from dynamics_models import *
from kf_measurement_models import *
from pf_measurement_models import *
from virtual_sensor_models import *
from utils import *
from filter_models import *

# Set origin position, orientation and timestamp
print("Setting origin position, orientation and timestamp")
origin_lla, origin_ecef = get_reference_from_gt("22 18 04.31949  114 10 44.60559        3.472")   
origin_rpy = get_reference_rot("-1.7398149928   0.4409540487 -132.534729738")
origin_time = 1621218775.00

# origin_lla, origin_ecef = get_reference_from_gt("22 18 00.25350  114 10 47.25217        3.189") 
# origin_rpy = get_reference_rot("-0.6671978933   0.4780759843 137.9359508957")
# origin_time = 1621218838.00


# origin_lla, origin_ecef = get_reference_from_gt("22 17 51.54462  114 10 37.53472        2.433")  
# origin_rpy = get_reference_rot("-0.7491022101  -0.4393383077 -133.811686506")
# origin_time = 1621218900.00

# origin_lla, origin_ecef = get_reference_from_gt("22 17 49.70559  114 10 33.84846        3.084") 
# origin_rpy = get_reference_rot("1.5505269260   0.5986893297 -84.4024418600")
# origin_time = 1621218930.00

# Set endtime
end_time = origin_time + 400 #+ 777.0

print("Loading data")
# 1. Load ground truth data
gt_pos, gt_vel, gt_acc, gt_rot, gt_len = load_ground_truth_select("/oak/stanford/groups/gracegao/HKdataset/data_06_22_22/UrbanNav_TST_GT_raw.txt", origin_lla)
# 2. Load dd data
dd_data = load_dd_data(origin_lla, origin_ecef, "/oak/stanford/groups/gracegao/Shubh/10_6_cleanup/KITTI360_Processing/TRI_KF/save_data/")
# 3. Load IMU data
imu_data = xr.DataArray(pd.read_csv("/oak/stanford/groups/gracegao/HKdataset/data_06_22_22/xsense_imu_medium_urban1.csv"))
# 4. Parse IMU data
timestamp, or_quat, or_cov, ang_vel, ang_vel_cov, lin_acc, lin_acc_cov = parse_imu_data(imu_data)
# 5. Load VO data
vo_data = prepare_vo_data("/scratch/users/shubhgup/1_18_winter/Left_features/2d3dmatches")
# 6. Load GPS Ambiguity and cycle slip data
# ints_data = get_ints_data()
# mixed_data = get_cycle_slip_data()

# Generate index converters
print("Generating index converters")
imu_to_gt_idx, imu_to_gnss_idx, utc_to_imu_idx, gt_to_imu_idx, utc_to_gt_idx, utc_to_gnss_idx, gt_idx_to_utc, gnss_idx_to_utc = gen_idx_converters(timestamp)
imu_to_vo_idx, vo_to_imu_idx = imu_to_vo_idx_from_timestamp(timestamp, vo_data)

# Generate ground truth deltas
gt_pos_delta, gt_rot_delta = gen_gt_deltas(gt_pos, gt_rot, imu_to_gt_idx, vo_to_imu_idx)

parameter_optimization_plots = False

if parameter_optimization_plots:
    # Load the results
    print("Loading the results... ", 'data/opt_state_dict_list.npy')
    opt_state_dict_list = np.load('data/opt_state_dict_list.npy', allow_pickle=True)

    # Save loss plot to image
    print("Saving loss plot to image")
    plt.figure()
    plot_loss(opt_state_dict_list, window_size=500)
    plt.savefig('data/loss_plot.png')

    param_names = ['r_std', 'p_std', 'y_std', 'gyr_bias_std', 'r_obs_std', 'p_obs_std', 'y_obs_std', 'vel_x_std', 'vel_y_std', 'speed_std', 'landmark_std', 'speed_scale', 'pos_x_std', 'pos_y_std', 'prange_std', 'carrier_std']

    # Save parameter trajectory to image
    print("Saving parameter trajectory to image")
    plt.figure()
    plot_parameters(opt_state_dict_list, names=param_names, window_size=500)
    plt.savefig('data/parameter_plot.png')

# Load recorded data from file
print("Loading recorded data... ", 'data/recorded_data.pkl')
with open('data/recorded_data.pkl', 'rb') as f:
    recorded_data = pickle.load(f)

N_dim, state_dim = create_state_dim(0, 16)

T_start, T, IMU_rate_div = get_imu_idx(origin_time, end_time, utc_to_imu_idx, 100)

print("Debug ", recorded_data['state_covariance'][0].shape)

estimated_states = torch.zeros(T, state_dim)
num_hypotheses = recorded_data['state_means'][0].shape[0]
state_means = torch.zeros(T, num_hypotheses, state_dim)
state_covariance = torch.eye(state_dim).reshape(1, 1, state_dim, state_dim).expand(T, num_hypotheses, state_dim, state_dim).detach().clone()
for i, t in enumerate(range(T_start+IMU_rate_div, T, IMU_rate_div)):
    estimated_states[t, :] = recorded_data['estimated_states'][i][0, :]
    state_means[t, :, :] = recorded_data['state_means'][i]
    state_covariance[t, :, :, :] = recorded_data['state_covariance'][i]

tmax = T

state_range = range(T_start, tmax, IMU_rate_div)
gt_range = [imu_to_gt_idx(t) for t in state_range]

# Save state trajectory to image
print("Saving state trajectory to image")
plt.figure()
plot_position_estimates(estimated_states, gt_pos, T_start, tmax, imu_to_gt_idx, IMU_rate_div)
plt.savefig('data/position_plot.svg')
plt.figure()
plot_trajectory(estimated_states, state_range, gt_pos, gt_range)
plt.savefig('data/trajectory_plot.svg')

# Save tracking error to image
print("Saving tracking error to image")
plt.figure()
plot_tracking_error(estimated_states, gt_pos, state_range, imu_to_gt_idx)
plt.savefig('data/tracking_error_plot.svg')

# Save position error bounds to image
print("Saving position error bounds to image")
plt.figure()
visualize_error_bounds(estimated_states, state_means, state_covariance, state_range, gt_pos, gt_rot, gt_range)
plt.savefig('data/error_bounds_plot.svg')

# Save rotation error to image
print("Saving rotation error to image")
plt.figure()
plot_orientation_estimates(estimated_states, state_range, gt_rot, gt_range)
plt.savefig('data/orientation_plot.svg')

# Save velocity error to image
print("Saving velocity error to image")
plt.figure()
plot_velocity_estimates(estimated_states, state_range, gt_pos, gt_range)
plt.savefig('data/velocity_plot.svg')
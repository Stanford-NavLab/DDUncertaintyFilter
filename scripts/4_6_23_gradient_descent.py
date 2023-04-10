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


print("Initializing filter")
N_dim, state_dim = create_state_dim(0, 16)

T_start, T, IMU_rate_div = get_imu_idx(origin_time, end_time, utc_to_imu_idx, 100)

dynamics_model = PosVelQuatBiasModel()

inter_const_bias = inter_const_bias_tensor(dd_data)

kf_measurement_model = init_filter_measurement_model(dd_data, N_dim, IMU_VO_DD_MeasurementModel)
# pf_measurement_model = init_filter_measurement_model(dd_data, N_dim, GNSSPFMeasurementModel_IMU_DD_VO)

reset_filter = gen_reset_function(state_dim, timestamp, gt_pos, gt_vel, gt_rot, imu_to_gt_idx, IMU_rate_div, T_start)

test_filter = AsyncExtendedKalmanFilter(
    dynamics_model=dynamics_model, # Initialise the filter with the dynamic model
    measurement_model=kf_measurement_model, # Initialise the filter with the measurement model
    )

# Reset filter
t = T_start + IMU_rate_div
recorded_data = reset_filter(test_filter, t - IMU_rate_div)

print("Initializing parameters")
dyn_parameters = recorded_data['dynamics_parameters'][-1]
obs_parameters = recorded_data['observation_parameters'][-1]

r_std_raw = torch.tensor(dyn_parameters['r_std'], requires_grad=False)
p_std_raw = torch.tensor(dyn_parameters['p_std'], requires_grad=False)
y_std_raw = torch.tensor(dyn_parameters['y_std'], requires_grad=False)
gyr_bias_std_raw = torch.tensor(dyn_parameters['gyr_bias_std'], requires_grad=False)

r_obs_std_raw = torch.tensor(obs_parameters['r_std'], requires_grad=False)
p_obs_std_raw = torch.tensor(obs_parameters['p_std'], requires_grad=False)
y_obs_std_raw = torch.tensor(obs_parameters['y_std'], requires_grad=False)

vel_x_std_raw = torch.tensor(dyn_parameters['vel_x_std'], requires_grad=True)
vel_y_std_raw = torch.tensor(dyn_parameters['vel_y_std'], requires_grad=True)

speed_std_raw = torch.tensor(obs_parameters['speed_std'], requires_grad=True)
landmark_std_raw = torch.tensor(obs_parameters['landmark_std'], requires_grad=True)
speed_scale_raw = torch.tensor(obs_parameters['speed_scale'], requires_grad=False)

pos_x_std_raw = torch.tensor(dyn_parameters['pos_x_std'], requires_grad=True)
pos_y_std_raw = torch.tensor(dyn_parameters['pos_y_std'], requires_grad=True)

prange_std_raw = torch.tensor(obs_parameters['prange_std'], requires_grad=True)
carrier_std_raw = torch.tensor(obs_parameters['carrier_std'], requires_grad=False)

param_dict_raw = { 'r_std': r_std_raw, 'p_std': p_std_raw, 'y_std': y_std_raw, 'gyr_bias_std': gyr_bias_std_raw, 'r_obs_std': r_obs_std_raw, 'p_obs_std': p_obs_std_raw, 'y_obs_std': y_obs_std_raw, 'vel_x_std': vel_x_std_raw, 'vel_y_std': vel_y_std_raw, 'speed_std': speed_std_raw, 'landmark_std': landmark_std_raw, 'speed_scale': speed_scale_raw, 'pos_x_std': pos_x_std_raw, 'pos_y_std': pos_y_std_raw, 'prange_std': prange_std_raw, 'carrier_std': carrier_std_raw}

optimizer = optim.AdamW(param_dict_raw.values(), lr=5e-2)

function_list = [None, None, None, None, 'softplus', 'softplus', 'softplus', 'softplus', 'softplus', 'softplus', 'softplus', None, 'softplus', 'softplus', 'softplus', 'softplus']
param_dict = get_param_dict(param_dict_raw, function_list)

opt_state_dict_list = []

imu_measurement_loss, vo_measurement_loss, vo_tight_measurement_loss, gnss_measurement_loss, supervised_position_loss = gen_measurement_losses(test_filter, jitter=1e-4)

print_params(param_dict)

try:
    print("Starting optimization")
    # k-step transition predict step gradient descent
    K_window = 10
    shuffled_times = list(range(T_start + IMU_rate_div, T - IMU_rate_div, IMU_rate_div))
    for epoch in tqdm(range(1000)):
        print("Epoch: ", epoch)
        np.random.shuffle(shuffled_times)
        for t_0 in tqdm(range(100), leave=False):
            t_0_shuffled = shuffled_times[t_0]
            recorded_data = reset_filter(test_filter, t_0_shuffled-IMU_rate_div)

            dyn_parameters = recorded_data['dynamics_parameters'][-1]
            obs_parameters = recorded_data['observation_parameters'][-1]
            prev_timestamp = timestamp[t_0_shuffled-IMU_rate_div]
            dd_idx = recorded_data['last_update_gnss']
            gt_idx = recorded_data['last_update_gt']
            vo_idx = recorded_data['last_update_vo']
            estimated_state_vo = None
            
            loss = 0.0
            
            for k in range(K_window):
                t = t_0_shuffled + k*IMU_rate_div
                # Load IMU data
                timestamp_t, or_quat_t, or_cov_t, ang_vel_t, ang_vel_cov_t, lin_acc_t, lin_acc_cov_t = timestamp[t], or_quat[t], or_cov[t], ang_vel[t], ang_vel_cov[t], lin_acc[t], lin_acc_cov[t]

                # Compute time difference
                dt = (timestamp_t - prev_timestamp)
                prev_timestamp = timestamp_t

                test_filter.update_dynamics(
                    dt=dt, 
                    pos_x_std=param_dict['pos_x_std'], 
                    pos_y_std=param_dict['pos_y_std'], 
                    pos_z_std=torch.tensor(dyn_parameters['pos_z_std']), 
                    vel_x_std=param_dict['vel_x_std'], 
                    vel_y_std=param_dict['vel_y_std'], 
                    vel_z_std=torch.tensor(dyn_parameters['vel_z_std']), 
                    r_std=param_dict['r_std'], 
                    p_std=param_dict['p_std'], 
                    y_std=param_dict['y_std'], 
                    acc_bias_std=torch.tensor(dyn_parameters['acc_bias_std']), 
                    gyr_bias_std=param_dict['gyr_bias_std']
                )

                estimated_state = imu_predict_and_update(test_filter, lin_acc_t, ang_vel_t, or_quat_t, param_dict['r_obs_std'], param_dict['p_obs_std'], param_dict['y_obs_std'], torch.tensor(obs_parameters['imu_robust_threshold']), m_estimation=False)
            
                loss += imu_measurement_loss(or_quat_t, estimated_state)*0.05
            
                # VO data
                new_vo_idx = imu_to_vo_idx(t)
                if new_vo_idx > vo_idx:
                    vo_idx = new_vo_idx

                    # Load VO data
                    landmark_3d, pixel_2d, K, ransac_R, ransac_t = load_vo_data(vo_idx, vo_data, size=50)
                    
                    estimated_state = vo_update(test_filter, estimated_state, landmark_3d, pixel_2d, K, ransac_R, ransac_t, param_dict['speed_std'], param_dict['speed_scale'], torch.tensor(obs_parameters['speed_robust_threshold']), vel_scaling_factor=IMU_rate_div/27/dt, m_estimation=False)
                    
                    if estimated_state_vo is not None:
                        estimated_state = vo_tight_update(
                            test_filter, estimated_state,
                            estimated_state_vo,
                            landmark_3d, pixel_2d, K, 
                            param_dict['landmark_std'], param_dict['speed_scale'],
                            torch.tensor(obs_parameters['speed_robust_threshold']),
                            vel_scaling_factor=IMU_rate_div/27/dt,
                            m_estimation=False
                            )
                        loss += vo_tight_measurement_loss(pixel_2d, estimated_state)*0.1
                    estimated_state_vo = estimated_state.detach().clone()

                    # loss += vo_measurement_loss(ransac_t, estimated_state)*0.1
                
                new_dd_idx = imu_to_gnss_idx(t)
                if new_dd_idx > dd_idx:
            #         print("GNSS update: ", t)
                    dd_idx = new_dd_idx

                    # Load GNSS observables
                    rover_code, base_code, rover_carr, base_carr, rover_cnos, satpos, idx_code_mask, idx_carr_mask = to_tensor(read_gnss_data(dd_data, dd_idx, 'mixed'))

                    gnss_observation, idx_code_mask, idx_carr_mask, ref = calc_gnss_observation(rover_code, base_code, rover_carr, base_carr, satpos, idx_code_mask, idx_carr_mask, ref=None, include_carrier=False)

                    estimated_state = gnss_update(test_filter, estimated_state, gnss_observation, satpos, ref, inter_const_bias, idx_code_mask, idx_carr_mask, param_dict['prange_std'], param_dict['carrier_std'], torch.tensor(obs_parameters['prange_robust_threshold']), m_estimation=False)
                    
                    loss += gnss_measurement_loss(gnss_observation, estimated_state)*0.2
            
            # Supervised update
            new_gt_idx = imu_to_gt_idx(t)
            if new_gt_idx > gt_idx:
                gt_idx = new_gt_idx
                loss += supervised_position_loss(gt_pos[gt_idx], estimated_state)
                
            if type(loss) != float: # and t_0%10==0:    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                opt_state_dict_list.append([param_dict[key].detach().numpy().copy() for key in param_dict.keys()] + [loss.detach().numpy().copy()])
                param_dict = get_param_dict(param_dict_raw, function_list)
        print("Loss ", loss.detach().numpy())
        print_params(param_dict)

# Save if keyboard interrupt
except KeyboardInterrupt:
    # Save the results
    print("Saving the results... ", 'data/opt_state_dict_list.npy')
    np.save('data/opt_state_dict_list.npy', opt_state_dict_list)

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
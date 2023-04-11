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
# 7. Load SLAM pose data
slam_data = load_slam_data("/scratch/users/shubhgup/1_18_winter/DDUncertaintyFilter/scripts/data/slam_data_gnss_imu_vo.pkl")

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
pf_measurement_model = init_filter_measurement_model(dd_data, N_dim, GNSSPFMeasurementModel_IMU_DD_VO)

reset_filter = gen_reset_function(state_dim, timestamp, gt_pos, gt_vel, gt_rot, imu_to_gt_idx, IMU_rate_div, T_start)

test_filter_base = AsyncExtendedKalmanFilter(
    dynamics_model=dynamics_model, # Initialise the filter with the dynamic model
    measurement_model=kf_measurement_model, # Initialise the filter with the measurement model
    )

test_filter = test_filter_base
# # Create an instance of the rao-blackwellized particle filter
# test_filter = AsyncRaoBlackwellizedParticleFilter(
#     dynamics_model=dynamics_model,  # Dynamics model
#     measurement_model=pf_measurement_model,  # Measurement model
#     resample=True,  # Resample particles
#     estimation_method="weighted_average",  # Use weighted average of particles
#     num_particles= 20,  # Number of particles
#     soft_resample_alpha=1.0,  # Soft resampling parameter
# )

# # mask for the position states
# pf_idx_mask = torch.zeros(state_dim, dtype=torch.bool)
# pf_idx_mask[:3] = True

# # This function attaches a filter to the test_filter object. It takes in the dynamics_model, measurement model, and index mask. The mode is set to 'linearization_points' by default. The function then assigns the dynamics model, measurement model, and index mask to the filter. The bank mode is used to specify whether the filter is a bank filter, which is a filter that uses multiple filters to handle multiple targets.

# test_filter.attach_ekf(test_filter_base, pf_idx_mask)

def run_timestep(t, recorded_data):
    
    dyn_parameters = recorded_data['dynamics_parameters'][-1]
    obs_parameters = recorded_data['observation_parameters'][-1]
    
    # Load IMU data
    timestamp_t, or_quat_t, or_cov_t, ang_vel_t, ang_vel_cov_t, lin_acc_t, lin_acc_cov_t = timestamp[t], or_quat[t], or_cov[t], ang_vel[t], ang_vel_cov[t], lin_acc[t], lin_acc_cov[t]

    # Compute time difference
    prev_timestamp = recorded_data['last_update_timestamp']
    dt = (timestamp_t - prev_timestamp)
    prev_timestamp = timestamp_t

    test_filter.update_dynamics(
            dt=dt, 
            pos_x_std=torch.tensor(dyn_parameters['pos_x_std']), 
            pos_y_std=torch.tensor(dyn_parameters['pos_y_std']), 
            pos_z_std=torch.tensor(dyn_parameters['pos_z_std']), 
            vel_x_std=torch.tensor(dyn_parameters['vel_x_std']), 
            vel_y_std=torch.tensor(dyn_parameters['vel_y_std']), 
            vel_z_std=torch.tensor(dyn_parameters['vel_z_std']), 
            r_std=torch.tensor(dyn_parameters['r_std']), 
            p_std=torch.tensor(dyn_parameters['p_std']), 
            y_std=torch.tensor(dyn_parameters['y_std']), 
            acc_bias_std=torch.tensor(dyn_parameters['acc_bias_std']), 
            gyr_bias_std=torch.tensor(dyn_parameters['gyr_bias_std'])
        )


    estimated_state = imu_predict_and_update(
        test_filter, 
        lin_acc_t, ang_vel_t, or_quat_t, 
        torch.tensor(obs_parameters['r_std']), torch.tensor(obs_parameters['p_std']), torch.tensor(obs_parameters['y_std']),
        torch.tensor(obs_parameters['imu_robust_threshold']),
        m_estimation=False
        )
    
    # VO data
    vo_idx = recorded_data['last_update_vo']
    new_vo_idx = imu_to_vo_idx(t)
    if new_vo_idx > vo_idx:
        vo_idx = new_vo_idx

        # Load VO data
        landmark_3d, pixel_2d, K, ransac_R, ransac_t = load_vo_data(vo_idx, vo_data, size=20)
        # if(np.linalg.norm(ransac_R) > 1e-1):
        #     print("---------------------------------------------------------------ransac_R: ", tf.matrix_to_euler_angles(torch.tensor(cv2.Rodrigues(ransac_R)[0]), ["X", "Y", "Z"]))

        prev_imu_idx = slam_data['time_to_index'](t - IMU_rate_div)
        if len(recorded_data['estimated_states']) > 0:
            # print("prev_imu_idx", prev_imu_idx)
            # print("slam_data['imu_times']", slam_data['imu_times'][prev_imu_idx])
            # print("len(recorded_data['estimated_states'])", len(recorded_data['estimated_states']))
            # prev_state = recorded_data['estimated_states'][prev_imu_idx]
            # # prev_state = recorded_data['estimated_states_vo'][len(recorded_data['estimated_states_vo'])-1]
            # prev_covariance = recorded_data['estimated_covariance'][prev_imu_idx]
        
            # # Provide GT locations of previous features
            # prev_gt_idx = imu_to_gt_idx(vo_to_imu_idx(vo_idx))-1
            # tmp_vel = (gt_pos[prev_gt_idx + 1, :] - gt_pos[prev_gt_idx, :])/400
            # prev_pos = gt_pos[prev_gt_idx, :] + tmp_vel*(vo_to_imu_idx(vo_idx) - gt_to_imu_idx(prev_gt_idx))
            # prev_state[:, :3] = torch.tensor(prev_pos)

            # Provide filter interpolated locations of previous features
            prev_state = slam_data['estimated_states'][prev_imu_idx]
            prev_covariance = slam_data['estimated_covariance'][prev_imu_idx]
            prev_t = slam_data['imu_times'][prev_imu_idx]
            tmp_vel = prev_state[0, 3:6]/400
            prev_pos = prev_state[0, :3] + tmp_vel*(vo_to_imu_idx(vo_idx) - prev_t)
            prev_state[:, :3] = prev_pos

            # print("prev_state: ", prev_state)
            # print("estimated_state (1): ", estimated_state)
            estimated_state = vo_tight_update(
                test_filter, estimated_state,
                prev_state, prev_covariance,
                landmark_3d, pixel_2d, K, 
                torch.tensor(obs_parameters['landmark_std']), torch.tensor(obs_parameters['speed_scale']),
                torch.tensor(obs_parameters['speed_robust_threshold']),
                vel_scaling_factor=IMU_rate_div/27/dt,
                m_estimation=False
                )
            # print("estimated_state (2): ", estimated_state)
        #         print("ransac_t: ", ransac_t)
        estimated_state = vo_update(
            test_filter, estimated_state,
            landmark_3d, pixel_2d, K, ransac_R, ransac_t, 
            torch.tensor(obs_parameters['speed_std']), torch.tensor(obs_parameters['speed_scale']),
            torch.tensor(obs_parameters['speed_robust_threshold']),
            vel_scaling_factor=IMU_rate_div/27/dt,
            m_estimation=False
            )

    # GNSS data
    dd_idx = recorded_data['last_update_gnss']
    new_dd_idx = imu_to_gnss_idx(t)
    if new_dd_idx > dd_idx:
        dd_idx = new_dd_idx

        # Load GNSS observables
        rover_code, base_code, rover_carr, base_carr, rover_cnos, satpos, idx_code_mask, idx_carr_mask = to_tensor(read_gnss_data(dd_data, dd_idx, 'mixed'))

        gnss_observation, idx_code_mask, idx_carr_mask, ref = calc_gnss_observation(rover_code, base_code, rover_carr, base_carr, satpos, idx_code_mask, idx_carr_mask, ref=None, include_carrier=False)

        estimated_state = gnss_update(
            test_filter, estimated_state, 
            gnss_observation, satpos, ref, inter_const_bias, idx_code_mask, idx_carr_mask, 
            torch.tensor(obs_parameters['prange_std']), torch.tensor(obs_parameters['carrier_std']), 
            torch.tensor(obs_parameters['prange_robust_threshold']),
            m_estimation=True
            )
    
    estimated_covariance = test_filter.get_covariance().detach().clone()
    
    # Update context
    recorded_data['last_update_timestamp'] = prev_timestamp
    recorded_data['last_update_imu'] = t
    if not recorded_data['last_update_vo'] == vo_idx:
        recorded_data['last_update_vo'] = vo_idx
    recorded_data['estimated_states'].append(estimated_state.detach().clone())
    recorded_data['estimated_covariance'].append(estimated_covariance.detach().clone())
    state_means, state_covariance = test_filter.get_state_statistics()
    recorded_data['state_means'].append(state_means.detach().clone())
    recorded_data['state_covariance'].append(state_covariance.detach().clone())
    recorded_data['imu_times'].append(t)
    
    return recorded_data

recorded_data = reset_filter(test_filter, T_start)

print("Running filter...")
print(test_filter)
with torch.no_grad():
    for t in tqdm(range(T_start+IMU_rate_div, T, IMU_rate_div)):
        recorded_data = run_timestep(t, recorded_data)

# Save recorded data to file
print("Saving recorded data... ", 'data/recorded_data.pkl')
with open('data/recorded_data.pkl', 'wb') as f:
    pickle.dump(recorded_data, f)

estimated_states = torch.zeros(T, state_dim)
num_hypotheses = recorded_data['state_means'][0].shape[0]
state_means = torch.zeros(T, num_hypotheses, state_dim)
state_covariance = torch.eye(state_dim).reshape(1, 1, state_dim, state_dim).expand(T, num_hypotheses, state_dim, state_dim)
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
# IMU only filter

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


    # Predict step
    controls = torch.cat((lin_acc_t, ang_vel_t)).float()
    # Axis fixing
    imu_observation = ahrs_meas_converter(or_quat_t.detach().clone()).float()
    
    test_filter.update_imu(
            r_std=torch.tensor(obs_parameters['r_std']),
            p_std=torch.tensor(obs_parameters['p_std']), 
            y_std=torch.tensor(obs_parameters['y_std']),
        )
    
    estimated_state = test_filter(controls=controls[None, :], observations=imu_observation[None, :])
    
    # Update context
    recorded_data['last_update_timestamp'] = prev_timestamp
    recorded_data['last_update_imu'] = t
    recorded_data['estimated_states'].append(estimated_state)
    
    return recorded_data
# estimated_state[:, 3:6].detach().numpy() - gt_pos_delta(vo_idx), gt_pos[imu_to_gt_idx(t)] - estimated_state[0, :3].detach().numpy(), eul2quat(torch.tensor(gt_rot[imu_to_gt_idx(t)])) - estimated_state[0, 6:10].detach().numpy()


# Run filter
with torch.no_grad():
    for t in tqdm(range(T_start+IMU_rate_div, T, IMU_rate_div)):
        recorded_data = run_timestep(t, recorded_data)

#---------------------------------------------------------------------------------------        
        
# Gradient descent config
dyn_parameters = recorded_data['dynamics_parameters'][-1]
obs_parameters = recorded_data['observation_parameters'][-1]
r_std = torch.tensor(dyn_parameters['r_std'], requires_grad=True)
p_std=torch.tensor(dyn_parameters['p_std'], requires_grad=True)
y_std=torch.tensor(dyn_parameters['y_std'], requires_grad=True)
gyr_bias_std = torch.tensor(dyn_parameters['gyr_bias_std'], requires_grad=True)

r_obs_std = torch.tensor(obs_parameters['r_std'], requires_grad=True)
p_obs_std=torch.tensor(obs_parameters['p_std'], requires_grad=True)
y_obs_std=torch.tensor(obs_parameters['y_std'], requires_grad=True)

optimizer = optim.Adam([y_std, gyr_bias_std, r_obs_std, p_obs_std, y_obs_std], lr=0.001)

y_std, gyr_bias_std, r_obs_std, p_obs_std, y_obs_std


# k-step transition predict step gradiet descent
K = 10
for t_0 in tqdm(range(T_start + IMU_rate_div, T - IMU_rate_div, IMU_rate_div)):
    recorded_data = reset_filter(test_filter, t_0-IMU_rate_div)
#     print(test_filter._belief_covariance[0, 6:10, 6:10])

    # print(test_filter._belief_mean)

    dyn_parameters = recorded_data['dynamics_parameters'][-1]
    obs_parameters = recorded_data['observation_parameters'][-1]
    prev_timestamp = timestamp[t_0-IMU_rate_div]
    
    loss = 0.0
    
    for k in range(K):
        t = t_0 + k*IMU_rate_div
        # Load IMU data
        timestamp_t, or_quat_t, or_cov_t, ang_vel_t, ang_vel_cov_t, lin_acc_t, lin_acc_cov_t = timestamp[t], or_quat[t], or_cov[t], ang_vel[t], ang_vel_cov[t], lin_acc[t], lin_acc_cov[t]

        # Compute time difference
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
            r_std=r_std, 
            p_std=p_std, 
            y_std=y_std, 
            acc_bias_std=torch.tensor(dyn_parameters['acc_bias_std']), 
            gyr_bias_std=gyr_bias_std
        )


        # IMU Predict and Update step
        controls = torch.cat((lin_acc_t, ang_vel_t)).float()
        
        # Axis fixing
        imu_observation = ahrs_meas_converter(or_quat_t.detach().clone()).float()
        test_filter.update_imu(
            r_std=r_obs_std,
            p_std=p_obs_std, 
            y_std=y_obs_std,
        )
        
        estimated_state = test_filter(controls=controls[None, :], observations=None)
#         print("Dynamics: ", test_filter._belief_covariance[0, 6:10, 6:10])
#         print("Observation: ", imu_observation)
#         print("Truth: ", eul2quat(torch.tensor(gt_rot[imu_to_gt_idx(t)])))
        
        dist = MultivariateNormal(
            estimated_state[0, 6:10], 
            covariance_matrix=1e-4*torch.eye(4) + test_filter._belief_covariance[0, 6:10, 6:10]
        )
        loss += -dist.log_prob(eul2quat(torch.tensor(gt_rot[imu_to_gt_idx(t)])))
        
        estimated_state = test_filter(controls=None, observations=imu_observation[None, :])
#         print("Update: ", estimated_state[0, 6:10] - eul2quat(torch.tensor(gt_rot[imu_to_gt_idx(t)])))
#         print("Update: ", test_filter._belief_covariance[0, 6:10, 6:10])

        # Supervised update
        dist = MultivariateNormal(
            estimated_state[0, 6:10], 
            covariance_matrix=1e-4*torch.eye(4) + test_filter._belief_covariance[0, 6:10, 6:10]
        )
        loss += -dist.log_prob(eul2quat(torch.tensor(gt_rot[imu_to_gt_idx(t)])))
        
        # Measurement likelihood update
        gt_state = estimated_state.detach().clone()
        gt_state[:, 6:10] = eul2quat(torch.tensor(gt_rot[imu_to_gt_idx(t)]))
        expected_obs, R_cholesky = test_filter.measurement_model(gt_state)        
        R = R_cholesky @ R_cholesky.transpose(-1, -2)
        dist = MultivariateNormal(expected_obs[0, :], covariance_matrix=1e-4*torch.eye(4) + R[0, :, :])
        loss += -dist.log_prob(imu_observation) * 0.1
       
    if (t_0//IMU_rate_div)%20==0:    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)
        print(r_std.grad, p_std.grad, y_std.grad)
        print(r_obs_std.grad, p_obs_std.grad, y_obs_std.grad)
        print(gyr_bias_std.grad)
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


    estimated_state = imu_predict_and_update(test_filter, lin_acc_t, ang_vel_t, or_quat_t, obs_parameters)
    
    # VO data
    vo_idx = recorded_data['last_update_vo']
    new_vo_idx = imu_to_vo_idx(t)
    if new_vo_idx > vo_idx:
        vo_idx = new_vo_idx

        # Load VO data
        landmark_3d, pixel_2d, K, ransac_R, ransac_t = load_vo_data(vo_idx, vo_data, size=50)

    #     # Load quaternion corresponding to previous image frame
    #     prev_frame_quat = or_quat[vo_to_imu_idx(vo_idx-1)].detach().clone()
    #     prev_frame_quat[[1, 2]] = prev_frame_quat[[2, 1]]
        # Compute change in orientation since previous frame
    #         delta_quat = tf.matrix_to_quaternion(torch.tensor(cv2.Rodrigues(ransac_R)[0]))
        delta_quat = tf.matrix_to_quaternion(torch.tensor(cv2.Rodrigues(np.zeros(3))[0]))

        # Update VO base model
        vel_scaling_factor = IMU_rate_div/27/dt
        vel_meas = torch.tensor([0.0, np.linalg.norm(ransac_t) * vel_scaling_factor, 0.0]).float()
#         vel_meas = torch.tensor(gt_vel[imu_to_gt_idx(t)]).float()

        test_filter.update_vo_base(std=torch.tensor(obs_parameters['speed_std']), scale=torch.tensor(obs_parameters['speed_scale']))
        estimated_state = test_filter(controls=None, observations=vel_meas[None, :])
    
    # Update context
    recorded_data['last_update_timestamp'] = prev_timestamp
    recorded_data['last_update_imu'] = t
    recorded_data['last_update_vo'] = vo_idx
    recorded_data['estimated_states'].append(estimated_state)
    
    return recorded_data
# estimated_state[:, 3:6].detach().numpy() - gt_pos_delta(vo_idx), gt_pos[imu_to_gt_idx(t)] - estimated_state[0, :3].detach().numpy(), eul2quat(torch.tensor(gt_rot[imu_to_gt_idx(t)])) - estimated_state[0, 6:10].detach().numpy()

# Run
with torch.no_grad():
    for t in tqdm(range(T_start+IMU_rate_div, T, IMU_rate_div)):
        recorded_data = run_timestep(t, recorded_data)
        
        
#-----------------------------------------------------------------------------------------------

dyn_parameters = recorded_data['dynamics_parameters'][-1]
obs_parameters = recorded_data['observation_parameters'][-1]

vel_x_std = torch.tensor(dyn_parameters['vel_x_std'], requires_grad=False)
vel_y_std = torch.tensor(dyn_parameters['vel_y_std'], requires_grad=False)

speed_std = torch.tensor(obs_parameters['speed_std'], requires_grad=False)
speed_scale = torch.tensor(obs_parameters['speed_scale'], requires_grad=True)

optimizer = optim.Adam([speed_scale], lr=0.01)

speed_scale

plt.plot([par[0] for par in opt_state_dict_list])
# plt.plot([par[1] for par in opt_state_dict_list])
# plt.plot([par[2] for par in opt_state_dict_list])

# k-step transition predict step gradiet descent
K_window = 10
opt_state_dict_list = []
for t_0 in tqdm(range(T_start + IMU_rate_div, T - IMU_rate_div, IMU_rate_div)):
    recorded_data = reset_filter(test_filter, t_0-IMU_rate_div)
#     print(test_filter._belief_covariance[0, 6:10, 6:10])

    # print(test_filter._belief_mean)

    dyn_parameters = recorded_data['dynamics_parameters'][-1]
    obs_parameters = recorded_data['observation_parameters'][-1]
    prev_timestamp = timestamp[t_0-IMU_rate_div]
    vo_idx = recorded_data['last_update_vo']
    
    loss = 0.0
    
    for k in range(K_window):
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
            vel_x_std=vel_x_std, 
            vel_y_std=vel_y_std, 
            vel_z_std=torch.tensor(dyn_parameters['vel_z_std']), 
            r_std=torch.tensor(dyn_parameters['r_std']), 
            p_std=torch.tensor(dyn_parameters['p_std']), 
            y_std=torch.tensor(dyn_parameters['y_std']), 
            acc_bias_std=torch.tensor(dyn_parameters['acc_bias_std']), 
            gyr_bias_std=torch.tensor(dyn_parameters['gyr_bias_std'])
        )

        estimated_state = imu_predict_and_update(test_filter, lin_acc_t, ang_vel_t, or_quat_t, obs_parameters)
        
        
        # VO data
        new_vo_idx = imu_to_vo_idx(t)
        if new_vo_idx > vo_idx:
            vo_idx = new_vo_idx

            # Load VO data
            landmark_3d, pixel_2d, K, ransac_R, ransac_t = load_vo_data(vo_idx, vo_data, size=50)
        
        #     # Load quaternion corresponding to previous image frame
        #     prev_frame_quat = or_quat[vo_to_imu_idx(vo_idx-1)].detach().clone()
        #     prev_frame_quat[[1, 2]] = prev_frame_quat[[2, 1]]
            # Compute change in orientation since previous frame
        #         delta_quat = tf.matrix_to_quaternion(torch.tensor(cv2.Rodrigues(ransac_R)[0]))
            delta_quat = tf.matrix_to_quaternion(torch.tensor(cv2.Rodrigues(np.zeros(3))[0]))
            
            # Update VO base model
            vel_scaling_factor = IMU_rate_div/27/dt
            vel_meas = torch.tensor([0.0, np.linalg.norm(ransac_t) * vel_scaling_factor, 0.0]).float()

            test_filter.update_vo_base(std=speed_std, scale=speed_scale)
            estimated_state = test_filter(controls=None, observations=vel_meas[None, :])
#             print("Estimated ", estimated_state[0, :6])
                                       
           # Supervised update
            dist = MultivariateNormal(
                estimated_state[0, :3], 
                covariance_matrix=1e-3*torch.eye(3) + test_filter._belief_covariance[0, :3, :3]
            )
#             print(test_filter._belief_covariance[0, 3:6, 3:6])
            loss += -dist.log_prob(torch.tensor(gt_pos[imu_to_gt_idx(t)]))
#             print("GT ", gt_pos[imu_to_gt_idx(t)], torch.tensor(gt_pos_delta(vo_idx))*vel_scaling_factor)

#             # Measurement likelihood update
#             gt_state = estimated_state.detach().clone()
#             gt_state[:, 3:6] = torch.tensor(gt_pos_delta(vo_idx))*vel_scaling_factor
#             expected_obs, R_cholesky = test_filter.measurement_model(gt_state) 
#             R = R_cholesky @ R_cholesky.transpose(-1, -2)
#             dist = MultivariateNormal(expected_obs[0, :], covariance_matrix=1e-3*torch.eye(3) + R[0, :, :])
# #             print(R[0, :, :])
#             loss += -dist.log_prob(vel_meas)
        
       
    if (t_0//IMU_rate_div)%1==0:    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        opt_state_dict_list.append([speed_scale.detach().numpy().copy()])

        print("Loss ", loss)
        print("Speed grad ", speed_scale)
#         print("Dynamics grads ", vel_x_std, vel_y_std)


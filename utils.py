import pandas as pd
import numpy as np
from threading import Thread
import cv2
import os
import time
import glob
import math
import torch
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
from itertools import chain, compress
import pytorch3d.transforms as tf
from queue import Queue
from coordinates import *
import pymap3d as pm
import xarray as xr

# quaternion representation: [x, y, z, w]
# JPL convention

####################################################################################################################
# HK dataset ops
####################################################################################################################

def read_lla(data_gt):
    return (dms2dec(data_gt[3:6]), dms2dec(data_gt[6:9]), data_gt['H-Ell'])

def read_rpy(data_gt):
    return (np.deg2rad(data_gt['Roll']), np.deg2rad(data_gt['Pitch']), np.deg2rad(90 + data_gt['Heading']))

def gt_to_pose(data_gt, H_body_cam, lcoord):
    R_body_world = torch.tensor(eul2rot(read_rpy(data_gt)), dtype=torch.float32)
    
    t_ned = lcoord.geodetic2ned(read_lla(data_gt))
#     t_enu = (t_ned[1], t_ned[0], t_ned[2])
    T_body_world = torch.tensor([*t_ned], dtype=torch.float32)
    
    H_body_world = vars_to_H(T_body_world, R_body_world)

    H_cam_world = H_body_world @ torch.linalg.inv(H_body_cam)
    return H_cam_world

def gt_to_lla(data_gt):
    return torch.tensor(read_lla(data_gt), dtype=torch.float32)

def gt_to_ecef(data_gt):
    return torch.tensor([*geodetic2ecef(read_lla(data_gt))], dtype=torch.float32)

def gt_to_ned(data_gt, lcoord):
    return torch.tensor([*lcoord.geodetic2ned(read_lla(data_gt))], dtype=torch.float32)

def get_reference_from_gt(line):
    vals = line.split("  ")
    reference_lla = [dms2dec(vals[0]), dms2dec(vals[1]), float(vals[-1])]
    reference_ecef = geodetic2ecef(reference_lla)
    return reference_lla, reference_ecef

def get_reference_rot(line):
    vals = line.split()
    reference_rpy = np.array([float(vals[0]), float(vals[1]), -float(vals[2])])
    return reference_rpy

def parse_imu_data(row):
    timestamp = (row.sel(dim_1="%time").to_numpy()/1e9)
    or_quat = row.sel(dim_1=['field.orientation.w', 'field.orientation.x', 'field.orientation.y', 'field.orientation.z']).to_numpy().astype(np.float32)
    or_cov = row.sel(dim_1=['field.orientation_covariance0', 'field.orientation_covariance1', 'field.orientation_covariance2', 
                      'field.orientation_covariance3','field.orientation_covariance4', 'field.orientation_covariance5',
                      'field.orientation_covariance6', 'field.orientation_covariance7', 'field.orientation_covariance8']).to_numpy().reshape((-1, 3, 3)).astype(np.float32)
    ang_vel = row.sel(dim_1=['field.angular_velocity.x', 'field.angular_velocity.y', 'field.angular_velocity.z']).to_numpy().astype(np.float32)
    ang_vel_cov = row.sel(dim_1=['field.angular_velocity_covariance0',
       'field.angular_velocity_covariance1',
       'field.angular_velocity_covariance2',
       'field.angular_velocity_covariance3',
       'field.angular_velocity_covariance4',
       'field.angular_velocity_covariance5',
       'field.angular_velocity_covariance6',
       'field.angular_velocity_covariance7',
       'field.angular_velocity_covariance8']).to_numpy().reshape((-1, 3, 3)).astype(np.float32)
    lin_acc = row.sel(dim_1=['field.linear_acceleration.x',
       'field.linear_acceleration.y', 'field.linear_acceleration.z']).to_numpy().astype(np.float32)
    lin_acc_cov = row.sel(dim_1=['field.linear_acceleration_covariance0',
       'field.linear_acceleration_covariance1',
       'field.linear_acceleration_covariance2',
       'field.linear_acceleration_covariance3',
       'field.linear_acceleration_covariance4',
       'field.linear_acceleration_covariance5',
       'field.linear_acceleration_covariance6',
       'field.linear_acceleration_covariance7',
       'field.linear_acceleration_covariance8']).to_numpy().reshape((-1, 3, 3)).astype(np.float32)
    return timestamp, torch.tensor(or_quat), torch.tensor(or_cov), torch.tensor(ang_vel), torch.tensor(ang_vel_cov), torch.tensor(lin_acc), torch.tensor(lin_acc_cov)

def gps2utc(time):
    return time - 95593 + 1621218775.0

def utc2gps(time):
    return time + 95593 - 1621218775

def read_gnss_data(dd_data, dd_tidx, constellation):
    if constellation=='mixed':
        dd_data_gps = read_gnss_data(dd_data, dd_tidx, 'gps')
        dd_data_beidou = read_gnss_data(dd_data, dd_tidx, 'beidou')
        rover_code, base_code, rover_carr, base_carr, rover_cnos, satpos, idx_code_mask, idx_carr_mask = tuple([np.concatenate([g, b], axis=0) for g, b in zip(dd_data_gps, dd_data_beidou)])
    elif constellation=='gps' or constellation=='beidou':
        rover_code = dd_data[constellation+'_rover_measurements_code'][dd_tidx].to_numpy()
        rover_cnos = dd_data[constellation+'_rover_measurements_cnos'][dd_tidx].to_numpy()
        base_code = dd_data[constellation+'_base_measurements_code'][dd_tidx].to_numpy()

        rover_carr = dd_data[constellation+'_rover_measurements_carr'][dd_tidx].to_numpy()
        base_carr = dd_data[constellation+'_base_measurements_carr'][dd_tidx].to_numpy()
        
        satpos = dd_data[constellation+'_enu_svs'][dd_tidx].to_numpy()

        idx_code_mask = np.logical_not(np.isnan(rover_code))
        idx_carr_mask = np.logical_not(np.isnan(rover_carr))
    return rover_code, base_code, rover_carr, base_carr, rover_cnos, satpos, idx_code_mask, idx_carr_mask

def load_ground_truth(fpath, origin_lla):
    fluff = 2
    header = ["UTCTime", "Week", "GPSTime", "Latitude", "Longitude", "H-Ell", "ECEFX", "ECEFY", "ECEFZ", "ENUX", "ENUY", "ENUZ", "VelBdyX", "VelBdyY", "VelBdyZ", "AccBdyX", "AccBdyY", "AccBdyZ", "Roll", "Pitch", "Heading", "Q"]
    all_data = []
    with open(fpath, "r") as f:
        for line in f:
            if fluff>0:
                fluff -= 1
                continue
            d = line.split()
            new_data = [float(d[0]), float(d[1]), float(d[2])]
            lla = [dms2dec(d[3:6]), dms2dec(d[6:9]), float(d[9])]
            ecef = list(geodetic2ecef(lla))
            enu = list(pm.geodetic2enu(*lla, *origin_lla))
            new_data += lla
            new_data += ecef
            new_data += enu
            new_data += [float(t) for t in d[10:]]
            all_data.append(new_data)
    all_data = np.array(all_data)
    return xr.DataArray(pd.DataFrame(all_data, columns=header))

def load_dd_data(origin_lla, x0):
    base_path = "/home/users/shubhgup/Codes/KITTI360_Processing/TRI_KF/save_data/"
    time_gt = gps2utc(np.load(os.path.join(base_path, "time_gt.npy")))
    
    base_station_ecef = np.array([-2414266.9197,5386768.9868, 2407460.0314])
    
    lat0 = origin_lla[0]
    lon0 = origin_lla[1]
    alt0 = origin_lla[2]
    
    base_station_enu = ecef2enu(base_station_ecef, lat0, lon0, x0)
    
    beidou_base_measurements_carr = np.load(os.path.join(base_path, "beidou_base_measurements_carr.npy"))
    beidou_base_measurements_code = np.load(os.path.join(base_path, "beidou_base_measurements_code.npy"))
    beidou_rover_measurements_carr = np.load(os.path.join(base_path, "beidou_rover_measurements_carr.npy"))
    beidou_rover_measurements_code = np.load(os.path.join(base_path, "beidou_rover_measurements_code.npy"))
    beidou_rover_measurements_cnos = np.load(os.path.join(base_path, "beidou_rover_measurements_cnos_new.npy"))
    beidou_ecef_svs = np.load(os.path.join(base_path, "beidou_ecef_svs.npy"))
    beidou_enu_svs = ecef2enu(beidou_ecef_svs, lat0, lon0, x0)
    
    gps_base_measurements_carr = np.load(os.path.join(base_path, "gps_base_measurements_carr.npy"))
    gps_base_measurements_code = np.load(os.path.join(base_path, "gps_base_measurements_code.npy"))
    gps_rover_measurements_carr = np.load(os.path.join(base_path, "gps_rover_measurements_carr.npy"))
    gps_rover_measurements_code = np.load(os.path.join(base_path, "gps_rover_measurements_code.npy"))
    gps_rover_measurements_cnos = np.load(os.path.join(base_path, "gps_rover_measurements_cnos_new.npy"))
    gps_ecef_svs = np.load(os.path.join(base_path, "gps_ecef_svs.npy"))
    gps_enu_svs = ecef2enu(gps_ecef_svs, lat0, lon0, x0)
    
    inter_const_bias = np.zeros(31 + 32)
    inter_const_bias[31:] = 17.5916
    
    return xr.Dataset(dict(
        time_gt=("t", time_gt), 
        beidou_base_measurements_carr=(["t", "sv_bei"], beidou_base_measurements_carr),
       beidou_base_measurements_code=(["t", "sv_bei"], beidou_base_measurements_code),
       beidou_rover_measurements_carr=(["t", "sv_bei"], beidou_rover_measurements_carr),
       beidou_rover_measurements_code=(["t", "sv_bei"], beidou_rover_measurements_code),
       beidou_rover_measurements_cnos=(["t", "sv_bei"], beidou_rover_measurements_cnos),
       beidou_enu_svs=(["t", "sv_bei", "pos"], beidou_enu_svs),
       gps_base_measurements_carr=(["t", "sv_gps"], gps_base_measurements_carr),
       gps_base_measurements_code=(["t", "sv_gps"], gps_base_measurements_code),
       gps_rover_measurements_carr=(["t", "sv_gps"], gps_rover_measurements_carr),
       gps_rover_measurements_code=(["t", "sv_gps"], gps_rover_measurements_code),
       gps_rover_measurements_cnos=(["t", "sv_gps"], gps_rover_measurements_cnos),
       gps_enu_svs=(["t", "sv_gps", "pos"], gps_enu_svs),
       base_station_enu=("pos", base_station_enu),
       inter_const_bias=("sv_all", inter_const_bias) 
      ))

def get_N_hypotheses_true(utc_t, ints_data, isref=False):
    true_mixed_ints = ints_data['tmi']
    true_gps_ints = ints_data['tgi']
    ref_gps = ints_data['ref']
    
    gps_t = int(utc2gps(utc_t))
    key = str(float(gps_t))
    if not (key in true_gps_ints.keys() or key in true_mixed_ints.keys()):
        return None, None
    true_gps_ints_t = true_gps_ints[key]
    true_gps_ints_t = {int(key[1:-1]): 0.190293*int(true_gps_ints_t[key]) for key in true_gps_ints_t.keys()}
    true_ints_t = true_gps_ints_t
    
    if key in true_mixed_ints.keys():
        true_mixed_ints_t = true_mixed_ints[key]
        true_mixed_ints_t = {int(key[1:-1]) + 31: 0.192*int(true_mixed_ints_t[key]) for key in true_mixed_ints_t.keys()}
        true_ints_t.update(true_mixed_ints_t)
    
    if isref:
        ref_gps_t = int(ref_gps[key])
    else:
        ref_gps_t = None
        
    return ref_gps_t, true_ints_t

def get_N_hypotheses(utc_t, ints_data, isref=False):
    mixed_ints = ints_data['mi']
    gps_ints = ints_data['gi']
    ref_gps = ints_data['ref']
    
    
    gps_t = int(utc2gps(utc_t))
    key = str(float(gps_t))
    if not (key in gps_ints.keys() or key in mixed_ints.keys()):
        return None, None
    
    gps_ints_t = gps_ints[key]
    
    if type(list(gps_ints_t.values())[0])==np.ndarray:
        gps_ints_t = {int(key[1:-1]): [0.190293*int(lkey) for lkey in gps_ints_t[key]] for key in  gps_ints_t.keys()}
    else:
        gps_ints_t = {int(key[1:-1]): [0.190293*int(gps_ints_t[key])] for key in gps_ints_t.keys()}
    
    ints_t =  gps_ints_t
    
    if key in mixed_ints.keys():
        mixed_ints_t = mixed_ints[key]
        if type(list(mixed_ints_t.values())[0])==np.ndarray:
            mixed_ints_t = {int(key[1:-1]) + 31: [0.192*int(lkey) for lkey in mixed_ints_t[key]] for key in  mixed_ints_t.keys()}
        else:
            mixed_ints_t = {int(key[1:-1]) + 31: [0.192*int(mixed_ints_t[key])] for key in mixed_ints_t.keys()}
        ints_t.update( mixed_ints_t)
    
    if isref:
        ref_gps_t = int(ref_gps[key])
    else:
        ref_gps_t = None
        
    return ref_gps_t, ints_t
####################################################################################################################
# SE(3) ops
####################################################################################################################

def vars_to_H(trans_vars, rot_vars):
    if torch.is_tensor(rot_vars):
        if torch.numel(rot_vars) == 3:
            rot_vars = eul2rot(rot_vars)
        return _vars_to_H_R(trans_vars, rot_vars)
    elif len(rot_vars)==2:
        return _vars_to_H_6dof(trans_vars, *rot_vars)
        
def compute_quat_dot(q, omega):
    real_parts = omega.new_zeros(omega.shape[:-1] + (1,))
    omega_as_quaternion = torch.cat((real_parts, omega), -1)
    return 0.5*tf.quaternion_raw_multiply(q, omega_as_quaternion)   


def _vars_to_H_R(trans_vars, rot_vars):
    """
    Convert translation nd rotation variables to transformation matrix in a differentiable way
    """
    one_hot_4 = torch.zeros((1, 4))
    one_hot_4[0, -1] = 1
    H_vo = torch.cat([rot_vars.reshape((3, 3)), trans_vars.reshape((3, 1))], 1)
    H_vo = torch.cat([H_vo, one_hot_4], 0)
    return H_vo

def _vars_to_H_6dof(trans_vars, v1_vars, v2_vars):
    """
    Convert translation nd rotation variables to transformation matrix in a differentiable way
    """
    one_hot_4 = torch.zeros((1, 4))
    one_hot_4[0, -1] = 1
    H_vo = torch.cat([sdof2rot(v1_vars, v2_vars), trans_vars.reshape((3, 1))], 1)
    H_vo = torch.cat([H_vo, one_hot_4], 0)
    return H_vo

def skew(vec):
    """
    Create a skew-symmetric matrix from a 3-element vector.
    """
    x, y, z = vec
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])

def to_rotation(q):
    """
    Convert a quaternion to the corresponding rotation matrix.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    q = q / np.linalg.norm(q)
    vec = q[:3]
    w = q[3]

    R = (2*w*w-1)*np.identity(3) - 2*w*skew(vec) + 2*vec[:, None]*vec
    return R

def to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    if R[2, 2] < 0:
        if R[0, 0] > R[1, 1]:
            t = 1 + R[0,0] - R[1,1] - R[2,2]
            q = [t, R[0, 1]+R[1, 0], R[2, 0]+R[0, 2], R[1, 2]-R[2, 1]]
        else:
            t = 1 - R[0,0] + R[1,1] - R[2,2]
            q = [R[0, 1]+R[1, 0], t, R[2, 1]+R[1, 2], R[2, 0]-R[0, 2]]
    else:
        if R[0, 0] < -R[1, 1]:
            t = 1 - R[0,0] - R[1,1] + R[2,2]
            q = [R[0, 2]+R[2, 0], R[2, 1]+R[1, 2], t, R[0, 1]-R[1, 0]]
        else:
            t = 1 + R[0,0] + R[1,1] + R[2,2]
            q = [R[1, 2]-R[2, 1], R[2, 0]-R[0, 2], R[0, 1]-R[1, 0], t]

    q = np.array(q) # * 0.5 / np.sqrt(t)
    return q / np.linalg.norm(q)

def quaternion_normalize(q):
    """
    Normalize the given quaternion to unit quaternion.
    """
    return q / np.linalg.norm(q)

def quaternion_conjugate(q):
    """
    Conjugate of a quaternion.
    """
    return np.array([*-q[:3], q[3]])

def quaternion_multiplication(q1, q2):
    """
    Perform q1 * q2
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    L = np.array([
        [ q1[3],  q1[2], -q1[1], q1[0]],
        [-q1[2],  q1[3],  q1[0], q1[1]],
        [ q1[1], -q1[0],  q1[3], q1[2]],
        [-q1[0], -q1[1], -q1[2], q1[3]]
    ])

    q = L @ q2
    return q / np.linalg.norm(q)

def eul2quat(eul):
    return tf.matrix_to_quaternion(tf.euler_angles_to_matrix(torch.deg2rad(torch.tensor(eul)), ["X", "Y", "Z"]))

def small_angle_quaternion(dtheta):
    """
    Convert the vector part of a quaternion to a full quaternion.
    This function is useful to convert delta quaternion which is  
    usually a 3x1 vector to a full quaternion.
    For more details, check Equation (238) and (239) in "Indirect Kalman 
    Filter for 3D Attitude Estimation: A Tutorial for quaternion Algebra".
    """
    dq = dtheta / 2.
    dq_square_norm = dq @ dq

    if dq_square_norm <= 1:
        q = np.array([*dq, np.sqrt(1-dq_square_norm)])
    else:
        q = np.array([*dq, 1.])
        q /= np.sqrt(1+dq_square_norm)
    return q

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta):
    if torch.is_tensor(theta):
        return _eul2rot_torch(theta)
    else:
        return _eul2rot_numpy(theta)
    
def sdof2rot(v1, v2):
    if torch.is_tensor(v1) and torch.is_tensor(v2):
        return _sdof2rot_torch(v1, v2)
    else:
        return _sdof2rot_numpy(v1, v2)

def _sdof2rot_torch(v1, v2):
    e1 = v1/torch.linalg.norm(v1)
    u2 = v2 - torch.dot(e1, v2)*e1
    e2 = u2/torch.linalg.norm(u2)
    R = torch.stack([e1, e2, torch.cross(e1, e2)])
    return R

def _sdof2rot_numpy(v1, v2):
    e1 = v1/np.linalg.norm(v1)
    u2 = v2 - np.dot(e1, v2)*e1
    e2 = u2/np.linalg.norm(u2)
    R = np.stack([e1, e2, np.linalg.cross(e1, e2)])
    return R
    
def _eul2rot_numpy(theta):

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

def _eul2rot_torch(theta):

    roll = torch.reshape(theta[0], (1,))
    yaw = torch.reshape(theta[1], (1,))
    pitch = torch.reshape(theta[2], (1,))

    tensor_0 = torch.zeros(1)
    tensor_1 = torch.ones(1)

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                    torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

    RY = torch.stack([
                    torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)

    RZ = torch.stack([
                    torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                    torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)

    return R

def quaternion_to_euler_angle_vectorized(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z 

def from_two_vectors(v0, v1):
    """
    Rotation quaternion from v0 to v1.
    """
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    d = v0 @ v1

    # if dot == -1, vectors are nearly opposite
    if d < -0.999999:
        axis = np.cross([1,0,0], v0)
        if np.linalg.norm(axis) < 0.000001:
            axis = np.cross([0,1,0], v0)
        q = np.array([*axis, 0.])
    elif d > 0.999999:
        q = np.array([0., 0., 0., 1.])
    else:
        s = np.sqrt((1+d)*2)
        axis = np.cross(v0, v1)
        vec = axis / s
        w = 0.5 * s
        q = np.array([*vec, w])
        
    q = q / np.linalg.norm(q)
    return quaternion_conjugate(q)   # hamilton -> JPL

# Utility transform functions
def to_4x4(H):
    return np.vstack([H, np.array([0, 0, 0, 1])])

def to_H(R, T):
    return to_4x4(np.hstack([R, T.reshape((-1, 1))]))

def stable_inverse(H):
    R = H[:3, :3]
    T = H[:3, 3:4]
    H_inv = np.hstack([R.T, -R.T @ T])
    return to_4x4(H_inv)

def flatten_H(H):
    return H.reshape(-1)[:-4]

def print_compare(a, b, transform=False):
    if transform:
        a = flatten_H(a)
        b = flatten_H(b)
    print(list(zip(a, b)))

class Isometry3d(object):
    """
    3d rigid transform.
    """
    def __init__(self, R, t):
        self.R = R
        self.t = t

    def matrix(self):
        m = np.identity(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    def inverse(self):
        return Isometry3d(self.R.T, -self.R.T @ self.t)

    def relative(self, T1):
        return T1 @ self.inverse()
    
    def __mul__(self, T1):
        R = self.R @ T1.R
        t = self.R @ T1.t + self.t
        return Isometry3d(R, t)

    def __str__(self):
        return "R: " + self.R.__str__() + "\nt: "+ self.t.__str__()

####################################################################################################################
# GNSS ops
####################################################################################################################    
    
def ecef2enu(x, lat0, lon0, x0, shift = True):
    x[x==0] = np.nan
    orig_shape = x.shape
    x = np.atleast_2d(x)
    
    phi = np.radians(lat0)
    lda = np.radians(lon0)

    sl = np.sin(lda)
    cl = np.cos(lda)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    if shift:
        x = x- np.tile(x0, (*orig_shape[:-1], 1))
        
    x_enu = -sl * x[..., 0] + cl * x[..., 1]
    y_enu = -cl * sp * x[..., 0] - sl * sp * x[..., 1] + cp * x[..., 2]
    z_enu = cl * cp * x[..., 0] + sl * cp * x[..., 1] + sp * x[..., 2]
    
    return np.stack((x_enu, y_enu, z_enu), -1).reshape(orig_shape)    
    
def expected_range(satXYZb, pos, idx_code_mask=None, idx_carr_mask=None):
    """
    satXYZb: (M, dim) [first 3 dims x y z]
    pos: (B, dim) [first 3 dims x y z]
    idx_code_mask, idx_carr_mask: (M, )
    """
    
    expected_observation_code = None
    expected_observation_carr = None
    
    if idx_code_mask is not None:
        expected_observation_code = batched_distance(satXYZb[idx_code_mask, :3], pos[:, :3])
    
    if idx_carr_mask is not None:
        expected_observation_carr = batched_distance(satXYZb[idx_carr_mask, :3], pos[:, :3])
        
    return expected_observation_code, expected_observation_carr

def expected_s_diff(satXYZb, pos, base_pose, idx_code_mask=None, idx_carr_mask=None, N_allsvs=None):
    """
    satXYZb: (M, dim) [first 3 dims x y z]
    pos: (B, dim) [first 3 dims x y z]
    base_pose: (B, dim) [first 3 dims x y z]
    idx_code_mask, idx_carr_mask: (M, )
    """
    
    M = satXYZb.shape[0]
    B = pos.shape[0]
    expected_observation_code = None
    expected_observation_carr = None
    
    rover_ranges_code, rover_ranges_carr = expected_range(satXYZb, pos, idx_code_mask=idx_code_mask, idx_carr_mask=idx_carr_mask)
    base_ranges_code, base_ranges_carr = expected_range(satXYZb, base_pose, idx_code_mask=idx_code_mask, idx_carr_mask=idx_carr_mask)
    
    if N_allsvs is None:
        N_allsvs = torch.zeros(B, M)
    
    if idx_code_mask is not None:
        expected_observation_code = rover_ranges_code - base_ranges_code      
    
    if idx_carr_mask is not None:
        expected_observation_carr = rover_ranges_carr - base_ranges_carr + N_allsvs[:, idx_carr_mask]
        
    return expected_observation_code, expected_observation_carr

def expected_d_diff(satXYZb, pos, base_pose, idx_code_mask=None, idx_carr_mask=None, ref_idx=0, inter_const_bias=None, N_allsvs=None):
    """
    satXYZb: (M, dim) [first 3 dims x y z]
    pos: (B, dim) [first 3 dims x y z]
    base_pose: (B, dim) [first 3 dims x y z]
    idx_code_mask, idx_carr_mask: (M, )
    ref_idx: Int
    """
    
    expected_observation_code = None
    expected_observation_carr = None
    
    M = satXYZb.shape[0]
    ref_mask = torch.zeros(M, dtype=torch.bool)
    ref_mask[ref_idx] = True
    
    s_diff_code, s_diff_carr = expected_s_diff(satXYZb, pos, base_pose, idx_code_mask=idx_code_mask, idx_carr_mask=idx_carr_mask, N_allsvs=N_allsvs)
    s_diff_code_ref, s_diff_carr_ref = expected_s_diff(satXYZb, pos, base_pose, idx_code_mask=ref_mask, idx_carr_mask=ref_mask, N_allsvs=N_allsvs)
    
    if idx_code_mask is not None:
        expected_observation_code = s_diff_code - s_diff_code_ref
        if inter_const_bias is not None:
            expected_observation_code += inter_const_bias[idx_code_mask]
    
    if idx_carr_mask is not None:
        expected_observation_carr = s_diff_carr - s_diff_carr_ref
        if inter_const_bias is not None:
            expected_observation_carr += inter_const_bias[idx_carr_mask]
        
    return expected_observation_code, expected_observation_carr

def compute_s_diff(rover_code, base_code, rover_carr, base_carr, idx_code_mask=None, idx_carr_mask=None):
    """
    rover_code, base_code, rover_carr, base_carr: (M, )
    idx_code_mask, idx_carr_mask: (M, )
    """
    
    sd_code = None
    sd_carr = None
    
    if idx_code_mask is not None:
        sd_code = rover_code - base_code
        sd_code = sd_code[idx_code_mask]
       
    if idx_carr_mask is not None:    
        sd_carr = rover_carr - base_carr
        sd_carr = sd_carr[idx_carr_mask]
        
    return sd_code, sd_carr

def compute_d_diff(rover_code, base_code, rover_carr, base_carr, idx_code_mask=None, idx_carr_mask=None, ref_idx=0):
    """
    rover_code, base_code, rover_carr, base_carr: (M, )
    idx_code_mask, idx_carr_mask: (M, )
    ref_idx: Int
    """
    
    dd_code = None
    dd_carr = None
    M = len(rover_code)
    
    assert M == len(base_code), "Rover and Base code have different lengths!"
    
    ones_mask = torch.ones(M, dtype=torch.bool)
    sd_code, sd_carr = compute_s_diff(rover_code, base_code, rover_carr, base_carr, idx_code_mask=ones_mask, idx_carr_mask=ones_mask)
    
    if idx_code_mask is not None:
        dd_code = sd_code - sd_code[ref_idx]
        dd_code = dd_code[idx_code_mask]

    if idx_carr_mask is not None:    
        dd_carr = sd_carr - sd_carr[ref_idx]
        dd_carr = dd_carr[idx_carr_mask]

    return dd_code, dd_carr

    
####################################################################################################################
# Matrix ops
####################################################################################################################

# (m X n), (b X n) -> (b X m)
def batched_mm(A, x):
    return (A[None, :, :] @ x[:, :, None]).squeeze(-1)

# (m X n), (b X n) -> (b X m)
def batched_distance(A, x):
    return torch.linalg.norm(A[None, :, :] - x[:, None, :], dim=-1)

####################################################################################################################
# Image ops
####################################################################################################################

# Helper vizualization function for showing opencv images
def show_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize= (15,8), dpi= 80)
    plt.imshow(img_rgb)

####################################################################################################################
# General ops
####################################################################################################################

def shift_bit_length(x):
    return 1<<((x-1).bit_length()-1)

def dms2dec(x):
    if type(x)==str:
        x = [float(x_i) for x_i in x.split(" ")]
    elif type(x[0])==str:
        x = [float(x_i) for x_i in x]
    return x[0] + x[1]/60 + x[2]/(60*60)

# Utility function for selecting arbitrary objects based on mask
def select(data, selectors):
    return [d for d, s in zip(data, selectors) if s]

# Stack several numpy objects along the last dimension and convert to tensor
def data_tensor(x_list):
    return torch.tensor(np.concatenate(tuple(x_list), -1), dtype=torch.float32)

def to_tensor(x_list):
    return tuple([torch.tensor(x) for x in x_list])

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val
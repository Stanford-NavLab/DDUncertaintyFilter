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
from torch.distributions.multivariate_normal import MultivariateNormal


# from dynamics_models import *
# from kf_measurement_models import *
# from pf_measurement_models import *
# from virtual_sensor_models import *
# from filter_models import *

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

# Define a function to load a dataset
def load_ground_truth_select(dataset_dir, origin_lla):
    """
    Load the dataset at a given directory
    """
    # Load the raw data from the dataset
    data = load_ground_truth(dataset_dir, origin_lla)

    # Process the data
    pos = data.sel(dim_1=["ENUX", "ENUY", "ENUZ"]).to_numpy()
    vel = data.sel(dim_1=["VelBdyX", "VelBdyY", "VelBdyZ"]).to_numpy()
    acc = data.sel(dim_1=["AccBdyX", "AccBdyY", "AccBdyZ"]).to_numpy()
    rot = data.sel(dim_1=["Roll", "Pitch", "Heading"]).to_numpy()
    
    return pos, vel, acc, rot, len(pos)

def load_dd_data(origin_lla, x0, base_path):
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

def calc_gnss_observation(rover_code, base_code, rover_carr, base_carr, satpos, idx_code_mask, idx_carr_mask, ref=None, include_carrier=True):
    # Check atleast 2 measurements in both code and carrier phase 
    if sum(idx_code_mask) < 2 or sum(idx_carr_mask) < 2:
        return (None, None, None, None)

    # First non-zero index is the reference
    if ref is None:
        ref = np.where(idx_code_mask & idx_carr_mask)[0][0]
#         ref = np.where(idx_code_mask[:31] & idx_carr_mask[:31])[0][-1]

    # Dont include reference in mask
    idx_code_mask[ref] = False
    idx_carr_mask[ref] = False

    # Compute double difference  
    dd_code, dd_carr = compute_d_diff(rover_code, base_code, rover_carr, base_carr, idx_code_mask=idx_code_mask, idx_carr_mask=idx_carr_mask, ref_idx=ref)
    
    if include_carrier:
        gnss_observation = data_tensor([dd_code, dd_carr])
    else:
        gnss_observation = data_tensor([dd_code])
    
    return gnss_observation, idx_code_mask, idx_carr_mask, ref

def load_vo_data(idx, vo_data, size=50, error_level=10.0):
    data = np.load(vo_data['3d2d_path'][idx])
    K = np.array([[264.9425,   0.    , 334.3975],
       [  0.    , 264.79  , 183.162 ],
       [  0.    ,   0.    ,   1.    ]]).astype(np.float32)
    landmark_3d = data['d3d_ref'][::2, :].astype(np.float32)
    pixel_2d = data['px_cur'][::2, :].astype(np.float32)
    _, cur_R, cur_t = cv2.solvePnP(landmark_3d, pixel_2d, K, np.zeros((4, 1)), rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    # Project 3D points to image plane
    img_pts, _ = cv2.projectPoints(landmark_3d, cur_R, cur_t, K, np.zeros((4, 1)))
    img_pts = img_pts.reshape(-1, 2)
    # Calculate reprojection error
    error = np.linalg.norm(pixel_2d - img_pts, axis=1)
    # Get inlier mask
    idx_mask = (error < error_level).nonzero()[0]
    
    np.random.shuffle(idx_mask)
    idx_mask = idx_mask[:size]
    
#     print(landmark_3d.shape)
    landmark_3d = torch.tensor(landmark_3d[idx_mask]).float()
#     print(landmark_3d.shape)
    pixel_2d = torch.tensor(pixel_2d[idx_mask]).float()
    K = torch.tensor(K)
    del data
    return landmark_3d, pixel_2d, K, cur_R, cur_t

# # Using straight velocity measurements
# def get_vo_data():
#     vo_path = os.path.abspath('/oak/stanford/groups/gracegao/HKdataset/data_06_22_22/')
#     vo_vals = np.load(os.path.join(vo_path, "velocity_vo_15Hz.npy"))
    
#     vo_times = np.load(os.path.join(vo_path, "velocity_vo_15Hz_timestamp.npy"))
    
#     return {
#         'vo': vo_vals.T,
#         'timestamp': vo_times/1e9
#     }

# Using matched 3d-2d features
def prepare_vo_data(input_path):
    # Get all the paths of 2d-3d matches
    paths_3d2d = sorted(glob.glob(input_path + "/*.npz"))
    # Get all the timestamps
    vo_times = np.array([float(os.path.basename(path)[:-4]) for path in paths_3d2d])
    
    return {
        '3d2d_path': paths_3d2d,
        'timestamp': vo_times/1e9
    }

def calc_gnss_cov(cnos, satpos, pos, mode='simple'):
    """Calculate the std deviation vector for GNSS measurements.

    Parameters
    ----------
    cnos : torch.Tensor
        A vector of CN/0 values in dB-Hz.
    satpos : torch.Tensor
        A matrix of satellite positions, with rows corresponding to satellites
        and columns corresponding to x, y, and z coordinates.
    pos : torch.Tensor
        A vector of the receiver position, with coordinates in x, y, and z.
    mode : str, optional
        The covariance model to use. Defaults to 'simple'.

    Returns
    -------
    torch.Tensor
        The std deviation vector for the GNSS measurements.

    """
    if mode == 'realni':
        los = satpos - pos[None, :]
        los = torch.div(los, torch.norm(los, dim=1)[:, None])
        el = torch.arccos(los[:, 2])

        s1 = 50
        A = 30
        s0 = 10 
        a = 50

        cno_term = cnos - s1
        cno_term_1 = torch.pow(10, -cno_term/a)
        cno_term_2 = cno_term/(s0-s1)
        s0_term = torch.pow(10, torch.tensor(-(s0-s1)/a))

        W = 1.0/torch.sin(el)**2*(cno_term_1*((A/s0_term - 1)*cno_term_2 + 1))
        
        sigma = 1./torch.sqrt(W)
    elif mode == 'simple':
        ksnr = 100
        
        sigma = ksnr*torch.pow(10, -cnos/20)
    
    return sigma

def calc_ambiguity_cov(cs_data, idx_carr_mask):
    """ 
    Create ambiguity std deviation vector.
    
    Parameters
    ----------
    cs_data : torch.Tensor
        1D array of True/False values indicating which ambiguities are cycle slip ambiguities.
    idx_carr_mask : torch.Tensor
        1D array of True/False values indicating which ambiguities are valid.
    
    Returns
    -------
    torch.Tensor
        1D array of the ambiguity std deviation vector.
    """
    default_sigma = torch.ones(len(idx_carr_mask))
    default_sigma[cs_data==True] = 20.0
    return default_sigma


def ahrs_meas_converter(quat):
    tmp_eul = quat2eul(tf.quaternion_invert(quat.detach()))
    tmp_eul = tmp_eul[[1, 0, 2]]
    tmp_eul[:2] = -tmp_eul[:2] 
    return eul2quat(tmp_eul, degree=False)
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

# Calculate delta quaternion between 2 quaternion vectors (torch)
def quat_delta(q1, q2):
    q1_inv = tf.quaternion_invert(q1)
    return tf.quaternion_multiply(q2, q1_inv)

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

def eul2quat(eul, degree=True):
    if degree:
        _eul = torch.deg2rad(eul)
    else:
        _eul = eul
    return tf.matrix_to_quaternion(tf.euler_angles_to_matrix(_eul, ["X", "Y", "Z"]))

def quat2eul(quat):
    return tf.matrix_to_euler_angles(tf.quaternion_to_matrix(quat), ["X", "Y", "Z"])

# Estimate quaternion between two vectors
def quat_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    q = np.array([R[2, 1], R[0, 2], R[1, 0], 1 + R[0, 0] + R[1, 1] + R[2, 2]])
    q = q / np.linalg.norm(q)
    return q

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

####################################################################################################################
# Index/timestamp syncing management
####################################################################################################################

def gen_utc_to_imu_idx(timestamp):
    min_utc = timestamp[0]
    max_utc = timestamp[-1]
    ls_utc_to_imu_idx = np.zeros(int(max_utc)+1-int(min_utc), dtype=np.int32)
    for i in range(len(timestamp)-1, 0, -1):
        ls_utc_to_imu_idx[int(timestamp[i])-int(min_utc)] = i 
    def utc_to_imu_idx(x):
        return ls_utc_to_imu_idx[int(x)-int(min_utc)]
    return utc_to_imu_idx

def gen_idx_converters(timestamp):
    def utc_to_gt_idx(utc):
        return int(utc - 1621218775.00)

    def gt_idx_to_utc(gt_idx):
        return gt_idx + 1621218775.00

    def utc_to_gnss_idx(utc):
        return int(utc - 1621218785)

    def gnss_idx_to_utc(gnss_idx):
        return gnss_idx + 1621218785

    def imu_to_gt_idx(i):
        return utc_to_gt_idx(timestamp[i])

    def imu_to_gnss_idx(i):
        return utc_to_gnss_idx(timestamp[i])

    def utc_to_imu_idx(utc):
        return gen_utc_to_imu_idx(timestamp)(utc)

    def gt_to_imu_idx(x):
        return utc_to_imu_idx(gt_idx_to_utc(x))
    
    return imu_to_gt_idx, imu_to_gnss_idx, utc_to_imu_idx, gt_to_imu_idx, utc_to_gt_idx, utc_to_gnss_idx, gt_idx_to_utc, gnss_idx_to_utc

def imu_to_vo_idx_from_timestamp(timestamp, vo_data):
    # Create a copy of the timestamp vector
    imu_to_vo_idx_data = np.zeros(len(timestamp), dtype=np.int32)
    vo_to_imu_idx_data = np.zeros(len(vo_data["timestamp"]), dtype=np.int32)

    # Initialize the index of the visual odometry data
    vo_idx = 0

    # Loop over all timestamps in the IMU data
    for i, t in enumerate(timestamp):
        # Loop over all timestamps in the visual odometry data until
        # we find a timestamp that is larger than the current one in
        # the IMU data
        while (t > vo_data["timestamp"][vo_idx]) and (vo_idx < len(vo_data["timestamp"])-1):
            vo_idx += 1

        # Store the index of the visual odometry data
        imu_to_vo_idx_data[i] = vo_idx
        
        # Store the index of the imu data
        vo_to_imu_idx_data[vo_idx] = i

    # Define the function to map the IMU index to the VO index
    def imu_to_vo_idx(imu_idx):
        return imu_to_vo_idx_data[imu_idx]
    
    # Define the function to map the IMU index to the VO index
    def vo_to_imu_idx(vo_idx):
        return vo_to_imu_idx_data[vo_idx]
    
    return imu_to_vo_idx, vo_to_imu_idx

####################################################################################################################
# GT ops
####################################################################################################################

def gen_gt_deltas(gt_pos, gt_rot, imu_to_gt_idx, vo_to_imu_idx):
    def gt_pos_delta(vo_idx):
        gt_diff = (gt_pos[imu_to_gt_idx(vo_to_imu_idx(vo_idx)), :] - gt_pos[imu_to_gt_idx(vo_to_imu_idx(vo_idx))-1, :])
        vo_idx_diff = 27
        gt_idx_diff = 400
        return gt_diff*vo_idx_diff/gt_idx_diff

    def gt_rot_delta(vo_idx):
        gt_diff = (gt_rot[imu_to_gt_idx(vo_to_imu_idx(vo_idx)), :] - gt_rot[imu_to_gt_idx(vo_to_imu_idx(vo_idx))-1, :])
        vo_idx_diff = 27
        gt_idx_diff = 400
        return gt_diff*vo_idx_diff/gt_idx_diff
    return gt_pos_delta, gt_rot_delta

####################################################################################################################
# Integer Ambiguity ops
####################################################################################################################

def get_ints_data():
    int_hypo_path = os.path.abspath('/home/users/shubhgup/Codes/KITTI360_Processing/TRI_KF/IntegerHypotheses')
    gps_ints = np.load(os.path.join(int_hypo_path, "true_gps_ints.npy"), allow_pickle=True).item()
    ref_gps = None # np.load("/home/users/shubhgup/Codes/KITTI360_Processing/TRI_KF/IntegerHypotheses/ref_gps.npy", allow_pickle=True).item()

    mixed_ints = np.load("/home/users/shubhgup/Codes/KITTI360_Processing/TRI_KF/IntegerHypotheses/true_mixed_ints.npy", allow_pickle=True).item()

    return {
        'mi': mixed_ints,
        'gi': gps_ints,
        'ref': ref_gps
    }

def get_cycle_slip_data():
    cycle_slip_path = os.path.abspath('/home/users/shubhgup/Codes/KITTI360_Processing/TRI_KF/cycle_slip_data/')
    _gps_cs = np.load(os.path.join(cycle_slip_path, 'gps mask_win5_thr3.npy'))
    _bei_cs = np.load(os.path.join(cycle_slip_path, 'bei mask_win5_thr3.npy'))

    mixed_cs = np.zeros((777, 31+32), dtype=np.bool)
    mixed_cs[5:, :31][_gps_cs==True] = True
    mixed_cs[5:, 31:][_bei_cs==True] = True
    return mixed_cs


####################################################################################################################
# Config
####################################################################################################################

def create_state_dim(N_dim=0, base_state_dim=16):
    state_dim = base_state_dim + N_dim
    return N_dim, state_dim

def get_imu_idx(origin_time, end_time, utc_to_imu_idx, IMU_rate_div=100):
    """Find the IMU indices of the origin and end times"""

    # Find the index of the origin time in the IMU data
    T_start = utc_to_imu_idx(origin_time)

    # Find the index of the end time in the IMU data
    T = utc_to_imu_idx(end_time)

    return T_start, T, IMU_rate_div

# Create a tensor of the inter-constellation bias
def inter_const_bias_tensor(dd_data):
    return torch.tensor(dd_data['inter_const_bias'].to_numpy())

def init_filter_measurement_model(dd_data, N_dim, meas_model):
    return meas_model(dd_data['base_station_enu'].to_numpy(), N_dim=N_dim)

def gen_reset_function(state_dim, timestamp, gt_pos, gt_vel, gt_rot, imu_to_gt_idx, IMU_rate_div, T_start):
    def reset_filter(test_filter, t=T_start):
        # initialize the previous timestamp
        prev_timestamp = timestamp[t] - 0.0025*IMU_rate_div
        
        init_state = torch.zeros(state_dim)
        init_state[:3] = torch.tensor(gt_pos[imu_to_gt_idx(t)])
        init_state[3:6] = torch.tensor(gt_vel[imu_to_gt_idx(t)])
        # init_state[3:6] = torch.zeros(3)
        init_state[6:10] = eul2quat(torch.tensor(gt_rot[imu_to_gt_idx(t)]))
    #     init_state[6:10] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        init_state[10:13] = torch.tensor([0, 0, 9.81])

        init_cov = torch.diag(torch.tensor([3.0, 3.0, 3.0, 
                            0.2, 0.2, 0.01,
                            0.1, 0.1, 0.1, 0.1,
                            1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0]))
        
        # Initialize the belief at the first time step
        test_filter.initialize_beliefs(
            mean=init_state[None, :].detach(), # (1, 4)
            covariance=init_cov[None, :, :].detach(), # (1, 4, 4)
        )
        # Create a data structure to hold the recorded data
        recorded_data = {
            'dynamics_parameters': [], # list of adaptive dynamics parameters
            'observation_parameters': [], # list of adaptive observation parameters
            'controls': [],  # List of control inputs
            'imu_observation': [],  # List of IMU observations
            'gnss_observation': [],  # List of GNSS observations
            'gnss_observation_context': [],  # List of GNSS observation contexts
            'last_update_timestamp': prev_timestamp,  # Last time data was added to the data structure
            'last_update_imu': -1,  # Last IMU idx data was added to the data structure
            'last_update_vo': -1,  # Last VO idx data was added to the data structure
            'last_update_gnss': -1,  # Last GNSS idx data was added to the data structure
            'last_update_gt': -1,  # Last GT idx data was added to the data structure
            'estimated_states': [], # Tracked estimates of states
        }

        recorded_data['dynamics_parameters'].append(dict(
            pos_x_std=5.109628, 
            pos_y_std=5.436571, 
            pos_z_std=1e-3, 
            vel_x_std=0.313416, 
            vel_y_std=2.103889, 
            vel_z_std=1e-4, 
            r_std=np.deg2rad(0.1), 
            p_std=np.deg2rad(0.1), 
            y_std=np.deg2rad(15.0), 
            acc_bias_std=1e-2, 
            gyr_bias_std=np.deg2rad(3.0)
        ))
        
        recorded_data['observation_parameters'].append(dict(
            r_std=0.0010, 
            p_std=0.0012, 
            y_std=0.236015,
            imu_robust_threshold=0.7,
            speed_std=0.703181,
            speed_scale=0.6006,
            speed_robust_threshold=5.0,
            prange_std=3.139834, 
            prange_robust_threshold=3.0,
            carrier_std=10.0,
        ))

        return recorded_data
    return reset_filter

####################################################################################################################
# Filter create ops (to fix)
####################################################################################################################

# # Create a particle filter
# test_filter = AsyncParticleFilter(
#     dynamics_model=dynamics_model, # The model for the particle filter to use for predicting the next state of the system
#     measurement_model=pf_measurement_model, # The model for the particle filter to use for updating the state of the system
#     resample=True, # Whether or not to resample the particles after each update step
#     estimation_method="weighted_average", # The method to use to calculate the next state estimate
#     num_particles= 10000, # The number of particles to use
#     soft_resample_alpha=1.0, # The alpha value to use for the soft resample
# )

# test_filter = AsyncExtendedInformationFilter(
#     dynamics_model=dynamics_model,  # Initialise the filter with the dynamic model
#     measurement_model=kf_measurement_model,  # Initialise the filter with the measurement model  
# )

# # Create an instance of the rao-blackwellized particle filter
# test_filter = AsyncRaoBlackwellizedParticleFilter(
#     dynamics_model=dynamics_model,  # Dynamics model
#     measurement_model=pf_measurement_model,  # Measurement model
#     resample=True,  # Resample particles
#     estimation_method="weighted_average",  # Use weighted average of particles
#     num_particles= 20,  # Number of particles
#     soft_resample_alpha=1.0,  # Soft resampling parameter
# )

# # Create a filter object for the Bank-of-EKF mode (RBPF without resampling)
# test_filter = AsyncRaoBlackwellizedParticleFilter(
#     dynamics_model=dynamics_model,
#     measurement_model=pf_measurement_model,
#     estimation_method="weighted_average",
#     num_particles= 1,
#     soft_resample_alpha=1.0,
# )

# PARTICLE FILTER STUFF:

# # mask for the position states
# pf_idx_mask = torch.zeros(state_dim, dtype=torch.bool)
# pf_idx_mask[:3] = True

# # This function attaches a filter to the test_filter object. It takes in the dynamics_model, measurement model, and index mask. The bank_mode is set to False by default. The function then assigns the dynamics model, measurement model, and index mask to the filter. The bank mode is used to specify whether the filter is a bank filter, which is a filter that uses multiple filters to handle multiple targets.

# test_filter.attach_ekf(dynamics_model, kf_measurement_model, pf_idx_mask, bank_mode=False)

####################################################################################################################
# IMU ops
####################################################################################################################

def imu_predict_and_update(test_filter, lin_acc_t, ang_vel_t, or_quat_t, r_std, p_std, y_std, imu_robust_threshold, m_estimation=True):
    controls = torch.cat((lin_acc_t, ang_vel_t)).float()

    # Axis fixing
    imu_observation = ahrs_meas_converter(or_quat_t.detach().clone()).float()
    test_filter.update_imu(
        r_std=r_std, 
        p_std=p_std, 
        y_std=y_std, 
        linearization_point=imu_observation[None, :]
    )

    # IMU Predict
    estimated_state = test_filter(controls=controls[None, :], observations=None)

    # # M-estimation
    # if m_estimation:
    #     expected_obs, R_cholesky = test_filter.measurement_model(estimated_state.detach().clone())        
    #     residual = torch.linalg.norm(imu_observation[None, :] - expected_obs)
    #     # print("imu_residual ", residual)
    #     if residual > imu_robust_threshold:
    #         return estimated_state

    # IMU Update
    estimated_state = test_filter(controls=None, observations=imu_observation[None, :])
    
    return estimated_state

####################################################################################################################
# VO ops
####################################################################################################################

def vo_update(test_filter, estimated_state, landmark_3d, pixel_2d, K, ransac_R, ransac_t, speed_std, speed_scale, vo_robust_threshold, vel_scaling_factor=1.0, m_estimation=True):
    #     # Load quaternion corresponding to previous image frame
    #     prev_frame_quat = or_quat[vo_to_imu_idx(vo_idx-1)].detach().clone()
    #     prev_frame_quat[[1, 2]] = prev_frame_quat[[2, 1]]
    # Compute change in orientation since previous frame
    #         delta_quat = tf.matrix_to_quaternion(torch.tensor(cv2.Rodrigues(ransac_R)[0]))
    delta_quat = tf.matrix_to_quaternion(torch.tensor(cv2.Rodrigues(np.zeros(3))[0]))

    # Update VO base model
    vel_meas = torch.tensor([0.0, np.linalg.norm(ransac_t) * vel_scaling_factor, 0.0]).float()
    # vel_meas = torch.tensor(gt_vel[imu_to_gt_idx(t)]).float()

    test_filter.update_vo_base(std=speed_std, scale=speed_scale)
    
    # # M-estimation
    # if m_estimation:
    #     expected_obs, R_cholesky = test_filter.measurement_model(estimated_state.detach().clone())        
    #     # print("expected_obs, vel_meas ", expected_obs, vel_meas)
    #     residual = torch.linalg.norm(vel_meas[None, :] - expected_obs)
    #     # print("vo_residual ", residual)
    #     if residual > vo_robust_threshold:
    #         return estimated_state
        
    estimated_state = test_filter(controls=None, observations=vel_meas[None, :])
    
    return estimated_state


####################################################################################################################
# GNSS ops
####################################################################################################################

def gnss_update(test_filter, estimated_state, gnss_observation, satpos, ref, inter_const_bias, idx_code_mask, idx_carr_mask, prange_std, carrier_std, gnss_robust_threshold, m_estimation=True):
    # LARGE_VALUE = 1e5
    
    if gnss_observation is None:
        return estimated_state

    # Update satellite and other context data in the measurement model
    test_filter.update_gnss(
        satXYZb=satpos, 
        ref_idx=ref, 
        inter_const_bias=inter_const_bias, 
        idx_code_mask=idx_code_mask, 
        idx_carr_mask=idx_carr_mask, 
        prange_std=prange_std,
        carrier_std=carrier_std,
        linearization_point=gnss_observation[None, :]
        )

    # # M-estimation
    # if m_estimation:
    #     expected_obs, R_cholesky = test_filter.measurement_model(estimated_state.detach().clone())        
    #     residual = torch.abs(gnss_observation[None, :] - expected_obs)

    #     mean_residual = torch.mean(residual)
    #     std_residual = torch.std(residual)
        
    #     outlier_mask =  residual[0, :] > mean_residual + gnss_robust_threshold*std_residual
    #     prange_std_vec = torch.ones(len(gnss_observation))*prange_std
    #     carrier_std_vec = torch.ones(len(gnss_observation))*carrier_std
    #     prange_std_vec[outlier_mask] = LARGE_VALUE
    #     carrier_std_vec[outlier_mask] = LARGE_VALUE


        # # print('residual', residual)
        # # Update satellite and other context data in the measurement model
        # test_filter.update_gnss(
        #     satXYZb=satpos, 
        #     ref_idx=ref, 
        #     inter_const_bias=inter_const_bias, 
        #     idx_code_mask=idx_code_mask, 
        #     idx_carr_mask=idx_carr_mask, 
        #     prange_std=prange_std_vec,
        #     carrier_std=carrier_std_vec
        #     )

    # Update step
    estimated_state = test_filter(observations=gnss_observation[None, :], controls=None)
    
    return estimated_state

####################################################################################################################
# Loss functions
####################################################################################################################

# This function computes the loss of the measurement likelihood update
# It takes as input the quaternion represented by the orientation measurement, 
# the estimated state, the filter to be used, and an optional jitter value
# It returns the negative log-likelihood of the observation

def gen_measurement_losses(test_filter, jitter=1e-4):
    def imu_measurement_loss(or_quat_t, estimated_state):
        # Measurement likelihood update
        expected_obs, R_cholesky = test_filter.measurement_model(estimated_state)        
        R = R_cholesky @ R_cholesky.transpose(-1, -2)
        dist = MultivariateNormal(expected_obs[0, :], covariance_matrix=jitter*torch.eye(4) + R[0, :, :])

        # The loss is the negative log-likelihood of the observation
        return -dist.log_prob(ahrs_meas_converter(or_quat_t.detach().clone()).float())
    
    def vo_measurement_loss(ransac_t, estimated_state):
        # Measurement likelihood update
        expected_obs, R_cholesky = test_filter.measurement_model(estimated_state)     
        expected_t = torch.linalg.norm(expected_obs)
        sigma = R_cholesky[0, 1, 1]
        # The loss is the negative log-likelihood of the observation
        return -torch.distributions.Normal(expected_t, sigma).log_prob(torch.tensor(np.linalg.norm(ransac_t)))
    
    def gnss_measurement_loss(gnss_observation, estimated_state):
        if gnss_observation is None:
            return 0
        # Measurement likelihood update
        expected_obs, R_cholesky = test_filter.measurement_model(estimated_state)        
        R = R_cholesky @ R_cholesky.transpose(-1, -2)
        dist = MultivariateNormal(expected_obs[0, :], covariance_matrix=jitter*torch.eye(expected_obs.shape[1]) + R[0, :, :])

        # The loss is the negative log-likelihood of the observation
        return -dist.log_prob(gnss_observation)

    def supervised_position_loss(gt_pos, estimated_state):
        # Supervised update
        dist = MultivariateNormal(
            estimated_state[0, :3], 
            covariance_matrix=torch.eye(3))
    #             print(test_filter._belief_covariance[0, 3:6, 3:6])
        return -dist.log_prob(torch.tensor(gt_pos))

    return imu_measurement_loss, vo_measurement_loss, gnss_measurement_loss, supervised_position_loss

####################################################################################################################
# Uncertainty Modules
####################################################################################################################

def compute_tau_bisection(epsilon, diff_xh_xi, weights, init_tau=30.0):
    tau = init_tau
    
    for _ in range(20): 
        is_inside = diff_xh_xi < tau
        filter_integral = torch.sum(weights[is_inside])
        if filter_integral < 1 - epsilon:
            tau = 3/2 * tau
        else:
            tau = 1/2 * tau
            
    return tau

def compute_tau_empirical_sigma(epsilon, diff_xh_xi, weights):
    sigma = torch.sqrt(torch.sum(weights * torch.square(diff_xh_xi))) + 1e-6
    return torch.distributions.normal.Normal(loc=0.0, scale=sigma).icdf(torch.tensor(1-epsilon))

def compute_tau_sigma(epsilon, var, weights):
    sigma = torch.sqrt(torch.sum(weights * var)) + 1e-6
    return torch.distributions.normal.Normal(loc=0.0, scale=sigma).icdf(torch.tensor(1-epsilon))

def integrate_beliefs_1d(estimated_state, all_hypo_log_weights, all_hypo_states, all_hypo_cov, epsilon=0.01, dim=0):
    if len(all_hypo_states.shape)==2:
        all_hypo_states = all_hypo_states[None, :, :]
        all_hypo_log_weights = all_hypo_log_weights[None, :]
        all_hypo_cov = all_hypo_cov[None, :, :, :]
    
    num_hypo = all_hypo_log_weights.shape[0]
    
#     diff_xh_xi = torch.abs(estimated_state[dim] - all_hypo_states[:, :, dim])
    all_hypo_tau = torch.tensor([compute_tau_sigma(epsilon, all_hypo_cov[0, i, dim, dim], 1.0) for i in range(all_hypo_states.shape[1])])
    diff_xh_xi = torch.abs(estimated_state[dim] - all_hypo_states[:, :, dim]) + all_hypo_tau[None, :]
    print(all_hypo_tau, torch.sqrt(all_hypo_cov[:, :, dim, dim]))
#     diff_xh_xi = torch.abs(estimated_state[dim] - all_hypo_states[:, :, dim]) + torch.sqrt(all_hypo_cov[:, :, dim, dim])
    
    weights = torch.exp(all_hypo_log_weights)
    
#     diff_atau_xi = torch.abs(estimated_state[dim] + tau - all_hypo_states[:, :, dim])
#     diff_btau_xi = torch.abs(estimated_state[dim] - tau - all_hypo_states[:, :, dim])
    
#     diff_ptau_xi = torch.maximum(diff_atau_xi, diff_btau_xi)
#     diff_mtau_xi = torch.minimum(diff_atau_xi, diff_btau_xi)
    
    tau = compute_tau_empirical_sigma(epsilon, diff_xh_xi, weights)
#     tau = compute_tau_sigma(epsilon, all_hypo_cov[:, :, dim, dim], weights)

    #     sigma = 5.0 # all_hypo_cov[:, :, dim, dim]
    #     integral_xh_xi = torch.erf(diff_xh_xi/sigma)
    #     integral_ptau_xi = torch.erf(diff_ptau_xi/sigma)
    #     integral_mtau_xi = torch.erf(diff_mtau_xi/sigma)

    return tau

####################################################################################################################
# Visualization Ops
####################################################################################################################


def visualize_particle_distribution(recorded_data, timestep, sync_gt):
    get_error = lambda x, timestep: torch.norm(x[0, :, :3] - sync_gt[timestep], dim=-1) 

    # Initial -> Predict
    plt.hist(get_error(recorded_data[timestep][0]['initial states'], timestep).numpy(), color='r', alpha=0.5, label='initial errors')
    plt.hist(get_error(recorded_data[timestep][0]['predicted state'], timestep).numpy(), color='g', alpha=0.5, label='predict errors')
    plt.legend()
    plt.show()

    plt.bar(get_error(recorded_data[timestep][0]['initial states'], timestep).numpy(), torch.exp(recorded_data[timestep][0]['initial logwt'][0, :]).numpy(), width=0.3, color='r', alpha=0.3, label='initial weights')
    plt.bar(get_error(recorded_data[timestep][0]['predicted state'], timestep).numpy(), torch.exp(recorded_data[timestep][0]['predicted logwt'][0, :]).numpy(), width=0.3, color='g', alpha=0.3, label='predicted weights')
    plt.legend()
    plt.show()

    # Predict -> Update
    if recorded_data[timestep][1] is not None:
        plt.hist(get_error(recorded_data[timestep][1]['initial states'], timestep).numpy(), color='g', alpha=0.5, label='predict errors')
        plt.hist(get_error(recorded_data[timestep][1]['corrected state'], timestep).numpy(), color='b', alpha=0.5, label='correct errors')
        plt.legend()
        plt.show()
        
        plt.bar(get_error(recorded_data[timestep][1]['initial states'], timestep).numpy(), torch.exp(recorded_data[timestep][1]['initial logwt'][0, :]).numpy(), width=0.3, color='g', alpha=0.3, label='predicted weights')
        plt.bar(get_error(recorded_data[timestep][1]['corrected state'], timestep).numpy(), torch.exp(recorded_data[timestep][1]['corrected logwt'][0, :]).numpy(), width=0.3, color='b', alpha=0.3, label='corrected weights')
        plt.legend()
        plt.show()

def plot_position_estimates(estimated_states, gt_pos, T_start, tmax, imu_to_gt_idx, IMU_rate_div):
    state_range = range(T_start, tmax, IMU_rate_div)
    num_elem = len(state_range)

    lower_gt = imu_to_gt_idx(T_start)
    upper_gt = imu_to_gt_idx(tmax)

    gt_len = upper_gt-lower_gt

    states = estimated_states.detach()
    plt.plot(np.linspace(1, gt_len, num=num_elem), states[state_range, 0], "r", label="estimated_x")
    plt.plot(np.linspace(1, gt_len, num=num_elem), states[state_range, 1], "b", label="estimated_y")
    plt.plot(np.linspace(1, gt_len, num=num_elem), states[state_range, 2], "g", label="estimated_z")

    plt.plot(np.linspace(1, gt_len, num=gt_len), gt_pos[lower_gt:upper_gt, 0], "r--", label="gt_x")
    plt.plot(np.linspace(1, gt_len, num=gt_len), gt_pos[lower_gt:upper_gt, 1], "b--", label="gt_y")
    plt.plot(np.linspace(1, gt_len, num=gt_len), gt_pos[lower_gt:upper_gt, 2], "g--", label="gt_z")

    # plt.plot(gt_pos[lower_t_s-root_t_s:, 0], "r--", label="gt_x")
    # plt.plot(gt_pos[lower_t_s-root_t_s:, 1], "b--", label="gt_y")
    # plt.plot(gt_pos[lower_t_s-root_t_s:, 2], "g--", label="gt_z")

    plt.xlabel("time [s]")
    plt.ylabel("position [m]")

    plt.legend()

def plot_orientation_estimates(states, state_range, gt_rot, gt_range):
    state_or_eul = np.stack(list(quaternion_to_euler_angle_vectorized(states[state_range, 6], states[state_range, 7], states[state_range, 8], states[state_range, 9])), 1)
    viz_block = range(0, len(state_or_eul))
    
    plt.plot(state_or_eul[viz_block, 0], "r", label="estimated roll")
    plt.plot(state_or_eul[viz_block, 1], "b", label="estimated pitch")
    plt.plot(state_or_eul[viz_block, 2], "g", label="estimated yaw")
    
    plt.plot(gt_rot[gt_range, 0], "r--", label="gt roll")
    plt.plot(gt_rot[gt_range, 1], "b--", label="gt pitch")
    plt.plot(gt_rot[gt_range, 2], "g--", label="gt yaw")

    plt.xlabel("time [s]")
    plt.ylabel("orientation [deg]")

    plt.legend()

def plot_velocity_estimates(states, state_range, gt_pos, gt_range):
    state_vel = states[state_range, 3:6].numpy()

    viz_block_kHz = range(0, len(state_vel))
    plt.plot(np.linspace(0, 100, num=len(viz_block_kHz)), state_vel[viz_block_kHz, 0], "r", label="estimated vx")
    plt.plot(np.linspace(0, 100, num=len(viz_block_kHz)), state_vel[viz_block_kHz, 1], "b", label="estimated vy")
    plt.plot(np.linspace(0, 100, num=len(viz_block_kHz)), state_vel[viz_block_kHz, 2], "g", label="estimated vz")

    plt.plot(np.linspace(0, 100, num=len(gt_range)-1), np.ediff1d(gt_pos[gt_range, 0]), "r--", label="gt vx")
    plt.plot(np.linspace(0, 100, num=len(gt_range)-1), np.ediff1d(gt_pos[gt_range, 1]), "b--", label="gt vy")
    plt.plot(np.linspace(0, 100, num=len(gt_range)-1), np.ediff1d(gt_pos[gt_range, 2]), "g--", label="gt vz")

    plt.xlabel("time [s]")
    plt.ylabel("velocity [m/s]")

    plt.legend()

def plot_tracking_error(estimated_states, gt_pos, state_range, imu_to_gt_idx):
    gt_range = [imu_to_gt_idx(t) for t in state_range]

    tracking_error = torch.norm(estimated_states[state_range, :2] - gt_pos[gt_range, :2], dim=1)
    print("Mean tracking error: ", torch.mean(tracking_error))
    plt.figure()
    plt.plot(tracking_error)
    plt.xlabel("time [s]")
    plt.ylabel("error [m]")
    
    plt.figure()
    plt.hist(tracking_error.numpy().flatten())
    plt.xlabel("error [m]")
    plt.ylabel("count")

def visualize_ekf_covariance(test_filter, names=None):
    plt.imshow(test_filter._belief_covariance[0, :, :])
    # plt.imshow(test_filter._belief_covariance[0, :, :])
    plt.colorbar()
    # torch.linalg.cond(test_filter.ekf._belief_covariance[0, :3, :3])
    if names is not None:
        plt.xticks(range(len(names)), names, rotation=90)
        plt.yticks(range(len(names)), names)

def plot_trajectory(states, state_range, gt_pos, gt_range):
    plt.figure()
    plt.plot(states[state_range, 0], states[state_range, 1])
    plt.plot(gt_pos[gt_range, 0], gt_pos[gt_range, 1], "r--")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend(["estimated", "ground truth"])


def visualize_error_bounds(xh, gt, nul, eul, t_s, root_t_s, lower_t_s, upper_t_s, lower_gt, upper_gt):
    run_x_hat = torch.stack(xh)
    run_gt = torch.stack(gt)

    overall_pe = torch.norm((run_x_hat[:, :2]-run_gt[:, :2]), dim=1)
    overall_ul = torch.maximum(torch.stack(nul), torch.stack(eul))

    # mask = overall_ul>50.0

    # run_x_hat[mask] = np.nan
    # run_gt[mask] = np.nan
    # overall_pe[mask] = np.nan
    # overall_ul[mask] = 50.0

    # print(torch.sum((overall_ul<overall_pe)&(overall_pe>15.0))/len(overall_pe))

    def visualize_1d_error_bound(dim, title):
        plt.figure(figsize=(8, 8))
        plt.xlabel("Position Error [m]")
        plt.ylabel("Error bound [m]")
        plt.scatter(overall_pe, overall_ul, color='k', s=3)
        plt.plot([0, 50], [0, 50], 'r--')
        plt.xlim([0, 50])
        plt.ylim([0, 50])

        plt.fill_between(range(len(run_gt)), run_x_hat[:, dim] - overall_ul, run_x_hat[:, dim] + overall_ul, color='g', alpha=0.5, label=title)
        # plt.plot(run_x_hat[:, dim], 'r', label='State Estimate')
        plt.plot(run_gt[:, dim], 'r--', label='Ground Truth')


        plt.legend()

    visualize_1d_error_bound(0, "North Error Bound")
    visualize_1d_error_bound(1, "East Error Bound")
    visualize_1d_error_bound(2, "Up Error Bound")

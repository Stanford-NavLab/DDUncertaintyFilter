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
from utils import *
from image_reader import *

"""
Modified from the codebase for MSCKF (Multi-State Constraint Kalman Filter)
GitHub: https://github.com/uoip/stereo_msckf
"""
class ImageProcessor(object):
    """
    Detect and track features in image sequences.
    """
    def __init__(self, config, verbose=True):
        self.config = config

        # Indicate if this is the first image message.
        self.is_first_img = True
        self.verbose = verbose

        # ID for the next new feature.
        self.next_feature_id = 0

        # Feature detector
        self.detector = cv2.FastFeatureDetector_create(self.config.fast_threshold)

        # IMU message buffer.
        self.imu_msg_buffer = []

        # Previous and current images
        self.cam0_prev_img_msg = None
        self.cam0_curr_img_msg = None

        # Features in the previous and current image.
        # list of lists of FeatureMetaData
        self.prev_features = [[] for _ in range(self.config.grid_num)]  # Don't use [[]] * N
        self.curr_features = [[] for _ in range(self.config.grid_num)]

        # Number of features after each outlier removal step.
        # keys: before_tracking, after_tracking, after_matching, after_ransac
        self.num_features = defaultdict(int)

        # load config
        # Camera calibration parameters
        self.cam0_resolution = config.cam0_resolution   # vec2
        self.cam0_intrinsics = config.cam0_intrinsics   # vec4
        self.cam0_distortion_model = config.cam0_distortion_model     # string
        self.cam0_distortion_coeffs = config.cam0_distortion_coeffs   # vec4

        # Take a vector from cam0 frame to the IMU frame.
        self.T_cam0_imu = np.linalg.inv(config.T_imu_cam0)
        self.R_cam0_imu = self.T_cam0_imu[:3, :3]
        self.t_cam0_imu = self.T_cam0_imu[:3, 3]

    def camera_callback(self, camera_msg):
        """
        Callback function for the camera images.
        """
        start = time.time()
        self.cam0_curr_img_msg = camera_msg.cam0_msg
        
        # Detect features in the first frame.
        if self.is_first_img:
            self.initialize_first_frame()
            self.is_first_img = False
            # Draw results.
            # self.draw_features()
        else:
            # Track the feature in the previous image.
            t = time.time()
            self.track_features()
            if self.verbose:
                print('___track_features:', time.time() - t)
            t = time.time()

            # Add new features into the current image.
            self.add_new_features()
            if self.verbose:
                print('___add_new_features:', time.time() - t)
            t = time.time()
            self.prune_features()
            if self.verbose:
                print('___prune_features:', time.time() - t)
            t = time.time()
            # Draw results.
            # self.draw_features()
            if self.verbose:
                print('___draw_features:', time.time() - t)
            t = time.time()

        if self.verbose:
            print('===image process elapsed:', time.time() - start, f'({camera_msg.timestamp})')
        
        try:
            return self.publish()
        finally:
            self.cam0_prev_img_msg = self.cam0_curr_img_msg
            self.prev_features = self.curr_features
            
            # Initialize the current features to empty vectors.
            self.curr_features = [[] for _ in range(self.config.grid_num)]

    def initialize_first_frame(self, img=None):
        """
        Initialize the image processing sequence, which is basically detect 
        new features on the first set of stereo images.
        """
        if img is None:
            img = self.cam0_curr_img_msg.image
        grid_height, grid_width = self.get_grid_size(img)

        # Detect new features on the first image.
        new_features = self.detector.detect(img)

        # Group the features into grids
        grid_new_features = [[] for _ in range(self.config.grid_num)]

        for i in range(len(new_features)):
            cam0_point = new_features[i].pt
            response = new_features[i].response

            row = int(cam0_point[1] / grid_height)
            col = int(cam0_point[0] / grid_width)
            code = row*self.config.grid_col + col

            new_feature = FeatureMetaData()
            new_feature.response = response
            new_feature.cam0_point = cam0_point
            grid_new_features[code].append(new_feature)

        # Sort the new features in each grid based on its response.
        # And collect new features within each grid with high response.
        for i, new_features in enumerate(grid_new_features):
            for feature in sorted(new_features, key=lambda x:x.response, 
                reverse=True)[:self.config.grid_min_feature_num]:
                self.curr_features[i].append(feature)
                self.curr_features[i][-1].id = self.next_feature_id
                self.curr_features[i][-1].lifetime = 1
                self.next_feature_id += 1
    
    def track_features(self, img=None):
        """
        Tracker features on the newly received stereo images.
        """
        if img is None:
            img = self.cam0_curr_img_msg.image
        grid_height, grid_width = self.get_grid_size(img)

        # TODO: Compute a rough relative rotation which takes a vector 
        # from the previous frame to the current frame.
        cam0_R_p_c = np.identity(3)

        # Organize the features in the previous image.
        prev_ids = []
        prev_lifetime = []
        prev_cam0_points = []
        
        for feature in chain.from_iterable(self.prev_features):
            prev_ids.append(feature.id)
            prev_lifetime.append(feature.lifetime)
            prev_cam0_points.append(feature.cam0_point)
        prev_cam0_points = np.array(prev_cam0_points, dtype=np.float32)

        # Number of the features before tracking.
        self.num_features['before_tracking'] = len(prev_cam0_points)

        # Abort tracking if there is no features in the previous frame.
        if len(prev_cam0_points) == 0:
            return

        # Track features using LK optical flow method.
        curr_cam0_points = self.predict_feature_tracking(
            prev_cam0_points, cam0_R_p_c, self.cam0_intrinsics)

        curr_cam0_points, track_inliers, _ = cv2.calcOpticalFlowPyrLK(
            self.cam0_prev_img_msg.image, self.cam0_curr_img_msg.image,
            prev_cam0_points.astype(np.float32), 
            curr_cam0_points.astype(np.float32), 
            **self.config.lk_params)
            
        # Mark those tracked points out of the image region as untracked.
        for i, point in enumerate(curr_cam0_points):
            if not track_inliers[i]:
                continue
            if (point[0] < 0 or point[0] > img.shape[1]-1 or 
                point[1] < 0 or point[1] > img.shape[0]-1):
                track_inliers[i] = 0

        # Collect the tracked points.
        prev_tracked_ids = select(prev_ids, track_inliers)
        prev_tracked_lifetime = select(prev_lifetime, track_inliers)
        prev_tracked_cam0_points = select(prev_cam0_points, track_inliers)
        curr_tracked_cam0_points = select(curr_cam0_points, track_inliers)

        # Number of features left after tracking.
        self.num_features['after_tracking'] = len(curr_tracked_cam0_points)

        # TODO: Add outlier removal via RANSAC or other methods (map aiding)
        for i in range(len(prev_tracked_cam0_points)):
            row = int(curr_tracked_cam0_points[i][1] / grid_height)
            col = int(curr_tracked_cam0_points[i][0] / grid_width)
            code = row * self.config.grid_col + col

            grid_new_feature = FeatureMetaData()
            grid_new_feature.id = prev_tracked_ids[i]
            grid_new_feature.lifetime = prev_tracked_lifetime[i] + 1
            grid_new_feature.cam0_point = curr_tracked_cam0_points[i]
            prev_tracked_lifetime[i] += 1

            self.curr_features[code].append(grid_new_feature)
            
        # Compute the tracking rate.
        # prev_feature_num = sum([len(x) for x in self.prev_features])
        # curr_feature_num = sum([len(x) for x in self.curr_features])

    def add_new_features(self, curr_img=None):
        """
        Detect new features on the image to ensure that the features are 
        uniformly distributed on the image.
        """
        if curr_img is None:
            curr_img = self.cam0_curr_img_msg.image
        grid_height, grid_width = self.get_grid_size(curr_img)

        # Create a mask to avoid redetecting existing features.
        mask = np.ones(curr_img.shape[:2], dtype='uint8')

        for feature in chain.from_iterable(self.curr_features):
            x, y = map(int, feature.cam0_point)
            mask[y-3:y+4, x-3:x+4] = 0

        # Detect new features.
        new_features = self.detector.detect(curr_img, mask=mask)

        # Collect the new detected features based on the grid.
        # Select the ones with top response within each grid afterwards.
        new_feature_sieve = [[] for _ in range(self.config.grid_num)]
        for feature in new_features:
            row = int(feature.pt[1] / grid_height)
            col = int(feature.pt[0] / grid_width)
            code = row * self.config.grid_col + col
            new_feature_sieve[code].append(feature)

        new_features = []
        for features in new_feature_sieve:
            if len(features) > self.config.grid_max_feature_num:
                features = sorted(features, key=lambda x:x.response, 
                    reverse=True)[:self.config.grid_max_feature_num]
            new_features.append(features)
        new_features = list(chain.from_iterable(new_features))

        # Group the features into grids
        grid_new_features = [[] for _ in range(self.config.grid_num)]
        for i in range(len(new_features)):
            cam0_point = new_features[i].pt
            response = new_features[i].response

            row = int(cam0_point[1] / grid_height)
            col = int(cam0_point[0] / grid_width)
            code = row*self.config.grid_col + col

            new_feature = FeatureMetaData()
            new_feature.response = response
            new_feature.cam0_point = cam0_point
            grid_new_features[code].append(new_feature)

        # Sort the new features in each grid based on its response.
        # And collect new features within each grid with high response.
        for i, new_features in enumerate(grid_new_features):
            for feature in sorted(new_features, key=lambda x:x.response, 
                reverse=True)[:self.config.grid_min_feature_num]:
                self.curr_features[i].append(feature)
                self.curr_features[i][-1].id = self.next_feature_id
                self.curr_features[i][-1].lifetime = 1
                self.next_feature_id += 1

    def prune_features(self):
        """
        Remove some of the features of a grid in case there are too many 
        features inside of that grid, which ensures the number of features 
        within each grid is bounded.
        """
        for i, features in enumerate(self.curr_features):
            # Continue if the number of features in this grid does
            # not exceed the upper bound.
            if len(features) <= self.config.grid_max_feature_num:
                continue
            self.curr_features[i] = sorted(features, key=lambda x:x.lifetime, 
                reverse=True)[:self.config.grid_max_feature_num]

    def predict_feature_tracking(self, input_pts, R_p_c, intrinsics):
        """
        predictFeatureTracking Compensates the rotation between consecutive 
        camera frames so that feature tracking would be more robust and fast.
        Arguments:
            input_pts: features in the previous image to be tracked.
            R_p_c: a rotation matrix takes a vector in the previous camera 
                frame to the current camera frame. (matrix33)
            intrinsics: intrinsic matrix of the camera. (vec3)
        Returns:
            compensated_pts: predicted locations of the features in the 
                current image based on the provided rotation.
        """
        # Return directly if there are no input features.
        if len(input_pts) == 0:
            return []

        # Intrinsic matrix.
        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])
        H = K @ R_p_c @ np.linalg.inv(K)

        compensated_pts = []
        for i in range(len(input_pts)):
            p1 = np.array([*input_pts[i], 1.0])
            p2 = H @ p1
            compensated_pts.append(p2[:2] / p2[2])
        return np.array(compensated_pts, dtype=np.float32)
    
    def publish(self):
        """
        Publish the features on the current image including both the 
        tracked and newly detected ones.
        """
        curr_ids = []
        curr_cam0_points = []
        for feature in chain.from_iterable(self.curr_features):
            curr_ids.append(feature.id)
            curr_cam0_points.append(feature.cam0_point)
            
        # curr_cam0_points_undistorted = self.undistort_points(
        #     curr_cam0_points, self.cam0_intrinsics,
        #     self.cam0_distortion_model, self.cam0_distortion_coeffs)
        
        features = []
        for i in range(len(curr_ids)):
            fm = FeatureMeasurement()
            fm.id = curr_ids[i]
            fm.u0 = curr_cam0_points[i][0]
            fm.v0 = curr_cam0_points[i][1]
            # fm.u0 = curr_cam0_points_undistorted[i][0]
            # fm.v0 = curr_cam0_points_undistorted[i][1]
            features.append(fm)

        feature_msg = namedtuple('feature_msg', ['timestamp', 'features'])(
            self.cam0_curr_img_msg.timestamp, features)
        return feature_msg

    def rescale_points(self, pts1, pts2):
        """
        Arguments:
            pts1: first set of points.
            pts2: second set of points.
        Returns:
            pts1: scaled first set of points.
            pts2: scaled second set of points.
            scaling_factor: scaling factor
        """
        scaling_factor = 0
        for pt1, pt2 in zip(pts1, pts2):
            scaling_factor += np.linalg.norm(pt1)
            scaling_factor += np.linalg.norm(pt2)

        scaling_factor = (len(pts1) + len(pts2)) / scaling_factor * np.sqrt(2)

        for i in range(len(pts1)):
            pts1[i] *= scaling_factor
            pts2[i] *= scaling_factor

        return pts1, pts2, scaling_factor

    def undistort_points(self, pts_in, intrinsics, distortion_model, 
        distortion_coeffs, rectification_matrix=np.identity(3),
        new_intrinsics=np.array([1, 1, 0, 0])):
        """
        Arguments:
            pts_in: points to be undistorted.
            intrinsics: intrinsics of the camera.
            distortion_model: distortion model of the camera.
            distortion_coeffs: distortion coefficients.
            rectification_matrix:
            new_intrinsics:
        Returns:
            pts_out: undistorted points.
        """
        if len(pts_in) == 0:
            return []
        
        pts_in = np.reshape(pts_in, (-1, 1, 2))
        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])
        K_new = np.array([
            [new_intrinsics[0], 0.0, new_intrinsics[2]],
            [0.0, new_intrinsics[1], new_intrinsics[3]],
            [0.0, 0.0, 1.0]])

        if distortion_model == 'equidistant':
            pts_out = cv2.fisheye.undistortPoints(pts_in, K, distortion_coeffs,
                rectification_matrix, K_new)
        else:   # default: 'radtan'
            pts_out = cv2.undistortPoints(pts_in, K, distortion_coeffs, None,
                rectification_matrix, K_new)
        return pts_out.reshape((-1, 2))

    def distort_points(self, pts_in, intrinsics, distortion_model, 
            distortion_coeffs):
        """
        Arguments:
            pts_in: points to be distorted.
            intrinsics: intrinsics of the camera.
            distortion_model: distortion model of the camera.
            distortion_coeffs: distortion coefficients.
        Returns:
            pts_out: distorted points. (N, 2)
        """
        if len(pts_in) == 0:
            return []

        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])

        if distortion_model == 'equidistant':
            pts_out = cv2.fisheye.distortPoints(pts_in, K, distortion_coeffs)
        else:   # default: 'radtan'
            homogenous_pts = cv2.convertPointsToHomogeneous(pts_in)
            pts_out, _ = cv2.projectPoints(homogenous_pts, 
                np.zeros(3), np.zeros(3), K, distortion_coeffs)
        return pts_out.reshape((-1, 2))
    
    def get_grid_size(self, img):
        """
        # Size of each grid.
        """
        grid_height = int(np.ceil(img.shape[0] / self.config.grid_row))
        grid_width  = int(np.ceil(img.shape[1] / self.config.grid_col))
        return grid_height, grid_width

    def draw_features(self, img=None):
        if img is None:
            img = self.cam0_curr_img_msg.image
        img_kp = img.copy() 
        kp = []
        for feature in chain.from_iterable(self.curr_features):
            kp.append(cv2.KeyPoint(*feature.cam0_point, 1))
        cv2.drawKeypoints(img, kp, img_kp, color=(255,0,0))
        show_image(img_kp)
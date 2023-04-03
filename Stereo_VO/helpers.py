import pandas as pd
import numpy as np
import cv2
import os, sys
import time
import glob
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from itertools import chain, compress
from scipy.stats import chi2

import os, sys
sys.path.append('/scratch/users/shubhgup/1_18_winter/DDUncertaintyFilter/SuperPoint')

from sp_extractor import *

kMinNumFeature = 1500

lk_params = dict(winSize  = (31, 31), 
#                  maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

color = np.random.randint(0, 255, (2000, 3))

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, 
            k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = np.array([k1, k2, p1, p2, k3])
    
    def intrinsic_matrix(self):
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ])
    
class StereoPinholeCamera:
    def __init__(self, params_l, H_l, params_r, H_r):
        self.pinhole_model_l = PinholeCamera(*params_l)
        self.pinhole_model_r = PinholeCamera(*params_r)
        self.H_l = H_l
        self.H_r = H_r
    
    def projection_matrix(self, cam="l"):
        if cam=="l":
            P = np.matmul(self.pinhole_model_l.intrinsic_matrix(), self.H_l[:3, :])
        elif cam=="r":
            P = np.matmul(self.pinhole_model_r.intrinsic_matrix(), self.H_r[:3, :])
        else:
            raise "Unsupported argument!"
        
        return P
    
############################################################################################
# Detection and Tracking utils
############################################################################################
def detect_init_features(img, detectionEngine, mode="fast"):
    if mode=="fast":
        kp = detectionEngine.detect(img[:-100, :])
    elif mode=="grid":
        TILE_H = 20
        TILE_W = 20
        
        #20x10 (wxh) tiles for extracting less features from images 
        H,W, _ = img.shape
        kp = []
        idx = 0
        for y in range(0, H-100, TILE_H):
            for x in range(0, W, TILE_W):
                imPatch = img[y:y+TILE_H, x:x+TILE_W]
                keypoints = detectionEngine.detect(imPatch)
                for pt in keypoints:
                    pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

                if (len(keypoints) > 20):
                    keypoints = sorted(keypoints, key=lambda x: -x.response)
                    for kpt in keypoints[0:20]:
                        kp.append(kpt)
                else:
                    for kpt in keypoints:
                        kp.append(kpt)
    elif mode=="superpoint":
        pts, desc, heatmap = detectionEngine.run(img[:-100, :])
        kp = (pts, desc)
    return kp

def featureTracking(image_ref, image_cur, px_ref, new_px_ref=None, mode="basic", nn_thresh=0.7, trackingEngine=None):
    if mode=="basic":
        # pack keypoint 2-d coords into numpy array
        if not type(px_ref)==np.ndarray:
            trackPoints1 = np.zeros((len(px_ref), 1,2), dtype=np.float32)
            for i,kpt in enumerate(px_ref):
                trackPoints1[i,:,0] = kpt.pt[0]
                trackPoints1[i,:,1] = kpt.pt[1]

        trackPoints2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, trackPoints1, None, flags=cv2.MOTION_AFFINE, **lk_params)  #shape: [k,2] [k,1] [k,1]

        # separate points that were tracked successfully
        ptTrackable = np.where(st == 1, 1,0).astype(bool)
        trackPoints1_KLT = trackPoints1[ptTrackable, ...]
        trackPoints2_KLT = trackPoints2[ptTrackable, ...]
        trackPoints2_KLT = np.around(trackPoints2_KLT)

        # among tracked points take points within error measue
        error = 4
        errTrackablePoints = err[ptTrackable, ...]
        errThresholdedPoints = np.where(errTrackablePoints < error, 1, 0).astype(bool)
        trackPoints1_KLT = trackPoints1_KLT[errThresholdedPoints, ...]
        trackPoints2_KLT = trackPoints2_KLT[errThresholdedPoints, ...]
    elif mode=="superpoint":
        assert new_px_ref is not None
        if trackingEngine is None:
            trackingEngine = PointTracker(max_length=10, nn_thresh=nn_thresh)
            trackingEngine.update(*px_ref)
        trackingEngine.update(*new_px_ref)
        tracks = trackingEngine.get_tracks(min_length=1)
        tracks[:, 1] /= float(nn_thresh)
        trackPoints1_KLT, trackPoints2_KLT = trackingEngine.draw_tracks(tracks)
    
    return trackPoints1_KLT, trackPoints2_KLT, trackingEngine

############################################################################################
# Disparity utils
############################################################################################

def gen_disparity_displaced_pts(trackPoints1_KLT, trackPoints2_KLT, ImT1_disparityA):
    #compute right image disparity displaced points
    trackPoints1_KLT_L = trackPoints1_KLT
    trackPoints2_KLT_L = trackPoints2_KLT

    trackPoints1_KLT_R = np.copy(trackPoints1_KLT_L)
    selectedPointMap = np.zeros(trackPoints1_KLT_L.shape[0])

    disparityMinThres = 0.0
    disparityMaxThres = 100.0
    
    H, W = ImT1_disparityA.shape
    
    for i in range(trackPoints1_KLT_L.shape[0]):
        p1_y = int(trackPoints1_KLT_L[i,1])
        p1_x = int(trackPoints1_KLT_L[i,0])
        if p1_y >= H or p1_y < 0 or p1_x >= W or p1_x < 0:
                continue
        
        T1Disparity = ImT1_disparityA[p1_y, p1_x]
        
        if (T1Disparity > disparityMinThres and T1Disparity < disparityMaxThres):
            trackPoints1_KLT_R[i, 0] = trackPoints1_KLT_L[i, 0] - T1Disparity
            selectedPointMap[i] = 1

    selectedPointMap = selectedPointMap.astype(bool)
    trackPoints1_KLT_L_3d = trackPoints1_KLT_L[selectedPointMap, ...]
    trackPoints1_KLT_R_3d = trackPoints1_KLT_R[selectedPointMap, ...]
    trackPoints2_KLT_L_3d = trackPoints2_KLT_L[selectedPointMap, ...]
    
    return (trackPoints1_KLT_L_3d, trackPoints1_KLT_R_3d), trackPoints2_KLT_L_3d

def gen_disparity(imgL, imgR, disparityEngine): 
    max_disparity=16
    height, width, _ = imgL.shape
    
    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    
    grayL = np.power(grayL, 0.75).astype('uint8')
    grayR = np.power(grayR, 0.75).astype('uint8')
    
    disparity_UMat = disparityEngine.compute(cv2.UMat(grayL),cv2.UMat(grayR))
    disparity = cv2.UMat.get(disparity_UMat)
    
    speckleSize = math.floor((width * height) * 0.0005)
    maxSpeckleDiff = (8 * 16) # 128
    
    cv2.filterSpeckles(disparity, 0, speckleSize, maxSpeckleDiff)
    
    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.).astype(np.uint8)
    
#     _, mask = cv2.threshold(disparity_scaled,0, 1, cv2.THRESH_BINARY_INV)
#     mask[:,0:120] = 0
#     disparity_scaled = cv2.inpaint(disparity_scaled, mask, 2, cv2.INPAINT_NS)
    
    return disparity_scaled

############################################################################################
# Stereo utility functions
############################################################################################

def perform_3d_triangulation(trackPoints1_KLT_L_3d, trackPoints1_KLT_R_3d, Proj1, Proj2):
    # 3d point cloud triagulation

    numPoints = trackPoints1_KLT_L_3d.shape[0]
    d3dPointsT1 = np.ones((numPoints,3))
    
    for i in range(numPoints):
        #for i in range(1):
        pLeft = trackPoints1_KLT_L_3d[i,:]
        pRight = trackPoints1_KLT_R_3d[i,:]

        X = np.zeros((4,4))
        X[0,:] = pLeft[0] * Proj1[2,:] - Proj1[0,:]
        X[1,:] = pLeft[1] * Proj1[2,:] - Proj1[1,:]
        X[2,:] = pRight[0] * Proj2[2,:] - Proj2[0,:]
        X[3,:] = pRight[1] * Proj2[2,:] - Proj2[1,:]

        [u,s,v] = np.linalg.svd(X)
        v = v.transpose()
        vSmall = v[:,-1]
        vSmall /= vSmall[-1]

        d3dPointsT1[i, :] = vSmall[0:-1]
    #     print (X)
    #     print (vSmall)
    
    return d3dPointsT1

def detect_stereo_features(frame_l, frame_r, **kwargs):
    px_ref_l = detect_init_features(frame_l, **kwargs)
    px_ref_l, px_ref_r = featureTracking(frame_l, frame_r, px_ref_l)
    return (px_ref_l, px_ref_r)

############################################################################################
# VO functions
############################################################################################


############################################################################################
# Visualization functions
############################################################################################

def viz_tracks(px_1, px_2, img_2):
    # Create a mask image for drawing purposes
    mask = np.zeros_like(img_2)

    # draw the tracks
    for i, (new, old) in enumerate(zip(px_2, px_1)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(img_2, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    plt.imshow(img)
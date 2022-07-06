# TODO: prune imports
import pandas as pd
import numpy as np
from threading import Thread
import cv2
import os
import time
import glob
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
from itertools import chain, compress
from queue import Queue

# A reader class for sequential Images with passed list of filenames and timestamps
class ImageReader(object):
    def __init__(self, ids, timestamps, starttime=-float('inf')):
        self.ids = ids  # List of file paths
        self.timestamps = timestamps  # list of timestamps associated with each file
        self.starttime = starttime  # timestamps[0] <= starttime <= timestamps[-1]
        self.cache = dict()
        self.idx = 0

        self.field = namedtuple('img_msg', ['timestamp', 'image'])  

        self.ahead = 10   # 10 images ahead of current index
        self.wait = 1.5   # waiting time

        self.preload_thread = Thread(target=self.preload)   # Cool way of preloading some future images for speedup
        self.thread_started = False

    def read(self, path):
        return cv2.imread(path, -1)
        
    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.wait:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue
            
            for i in range(self.idx, self.idx + self.ahead):
                if self.timestamps[i] < self.starttime:
                    continue
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        # if not self.thread_started:
        #     self.thread_started = True
        #     self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:   
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            if timestamp < self.starttime:
                continue
            yield self.field(timestamp, self[i])

    def start_time(self):
        return self.timestamps[0]

    def set_starttime(self, starttime):
        self.starttime = starttime

# Class for holding feature data
class FeatureMetaData(object):
    """
    Contain necessary information of a feature for easy access.
    """
    def __init__(self):
        self.id = None           # int
        self.response = None     # float
        self.lifetime = None     # int
        self.cam0_point = None   # vec2

# Class for holding Optical Flow measurement (point-wise)
class FeatureMeasurement(object):
    """
    Stereo measurement of a feature.
    """
    def __init__(self):
        self.id = None
        self.u0 = None
        self.v0 = None


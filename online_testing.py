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

# Monocular camera object that simulates a real camera
class MonoCamera(object):
    def __init__(self, cam0):
        self.cam0 = cam0
        self.timestamps = cam0.timestamps

        self.field = namedtuple('camera_msg', 
            ['timestamp', 'cam0_image', 'cam0_msg'])

    def __iter__(self):
        for l in self.cam0:
            yield self.field(l.timestamp, l.image, l)

    def __len__(self):
        return len(self.cam0)

    def start_time(self):
        return self.cam0.starttime

    def set_starttime(self, starttime):
        self.starttime = starttime
        self.cam0.set_starttime(starttime)

# General sensor publisher class to simulate the real-time environment
class DataPublisher(object):
    def __init__(self, dataset, out_queue, duration=float('inf'), ratio=1.): 
        self.dataset = dataset
        self.dataset_starttime = dataset.starttime
        self.out_queue = out_queue
        self.duration = duration
        self.ratio = ratio
        self.starttime = None
        self.started = False
        self.stopped = False

        self.publish_thread = Thread(target=self.publish)
        
    def start(self, starttime):
        self.started = True
        self.starttime = starttime
        self.publish_thread.start()

    def stop(self):
        self.stopped = True
        if self.started:
            self.publish_thread.join()
        self.out_queue.put(None)

    def publish(self):
        dataset = iter(self.dataset)
        while not self.stopped:
            try:
                data = next(dataset)
            except StopIteration:
                self.out_queue.put(None)
                return

            interval = data.timestamp - self.dataset_starttime
            if interval < 0:
                continue
            while (time.time() - self.starttime) * self.ratio < interval + 1e-3:
                time.sleep(1e-3)   # assumption: data frequency < 1000hz
                if self.stopped:
                    return

            if interval <= self.duration + 1e-3:
                self.out_queue.put(data)
            else:
                self.out_queue.put(None)
                return


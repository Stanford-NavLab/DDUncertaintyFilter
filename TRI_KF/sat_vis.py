import numpy as np
import pandas as pd
from numpy import genfromtxt
from new_utils import *
import matplotlib.pyplot as plt


sat = np.load('beidou_sat.npy',allow_pickle=True)
time = np.load('tgps.npy',allow_pickle=True)  

print (len(time))
print (len(sat))

plt.plot(time,sat,'r.')
plt.xlabel('GPS Time (seconds)')
plt.ylabel('Number of sats (Beidou)')
plt.show()

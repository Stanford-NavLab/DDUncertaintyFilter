#!/usr/bin/env python3
from operator import le
import os
from textwrap import wrap
import numpy as np
import math

#from scipy.sparse.linalg.dsolve.linsolve import use_solver
from preprocessing import *
import matplotlib.pyplot as plt
import pandas as pd
from utils import *


np.random.seed(0)


dir_path = os.path.dirname(os.path.realpath(__file__))
#date = np.datetime64("2019-04-28")  #dataset 1
date = np.datetime64("2021-05-17")   #dataset 2

# params
leap_seconds = 18

x0 = np.array([-2414266.9197,5386768.9868, 2407460.0314])
#cartesian coordinates from https://www.geodetic.gov.hk/common/data/pdf/SatRef_Coord.pdf

#name_obs1 = dir_path + '/20210826_data/COM3_190428_124409.obs' 
#name_obs1 = dir_path + '/COM55_200314_084357.obs' #trying with the second .obs file
# name_obs2 = dir_path + '/20210826_data/hksc118m.19o'
# name_eph =  dir_path + '/20210826_data/hksc118m.19f'

#name_obs2 = dir_path + '/hksc074j.20o'
#name_eph =  dir_path + '/hksc074j.20f'


#phone dataset
name_obs1 = dir_path + '/medium_dataset/20210517.light-urban.tste.huawei.p40pro.obs' 
name_obs2 = dir_path + '/medium_dataset/hksc137c.21o'
name_eph1 = dir_path + '/medium_dataset/hksc137c.21n'
name_eph2 = dir_path + '/medium_dataset/hksc137c.21f'


traj1, traj2, gps_eph, bei_eph = loadTrajectories(name_obs1, name_obs2, name_eph1, name_eph2)
print('trajectories loaded')

t_gps, svs, code1, code2, carrier1, carrier2, ts = constructMeasurements_gps(traj1, traj2, date, sort_cn0 = True)
t_bei, svbs, codeb1, codeb2, carrierb1, carrierb2, tbs = constructMeasurements_bei(traj1, traj2, date, sort_cn0 = False)


# print ('time',t_gps)   

# print ('time',t_bei) 

# print ('gps sats', svs )
# print ('bei sats', svbs)

# print ('gps times', t_gps[0:5])
# print ('bei times', t_bei[0:5])

# print (t_gps)
# print ('codes', code1[0])
# # print ('carriers1', carrier1)
# print ('codesb', codeb1[0])
# print ('carriersb1', carrierb1)

f_gps = 1575.42 * 10 ** 6
f_bei = 1561.1 * 10 ** 6
c = 299792458
lda_gps = c / f_gps
lda_bei = c / f_bei


psi_all = []
psib_all = []
G_all = []
G_all_b = []
sat_pos_all = []
satb_pos_all = []
t_all =[]


new = np.zeros((8, len(t_gps),10))

for i in range(0, len(t_gps)):   
    if len(svs[i]) >= 2 and len(svbs[i]) >= 2:  #confirm with Shubh about this requirement.. need to exclude R satellites
        # print ('here',svbs[i])
        # print ('here2',svs[i])
        psi, psib,H, Hb, sat_pos, satb_pos = prepareData_mixed(round(t_gps[i]),svs[i], svbs[i],code1[i], code2[i], codeb1[i], codeb2[i], \
            carrier1[i], carrier2[i], carrierb1[i], carrierb2[i], gps_eph, bei_eph, lda_gps, lda_bei,\
                 plane=False, ref= -1, x0=x0)  #returns everything in ECEF coordinates
        # print ('p',psi)
        #print ('pb',psib)
        # print ('H',H)
        # print ('Hb',Hb)
        # print ('satb',satb_pos)
        new[0,i, :len(code1[i])] = code1[i]
        new[1,i, :len(code2[i])] = code2[i]
        new[2,i,:len(codeb1[i])] = codeb1[i]
        new[3,i,:len(codeb2[i])] = codeb2[i]

        new[4,i,:len(carrier1[i])] = carrier1[i]
        new[5,i,:len(carrier2[i])] = carrier2[i]
        new[6,i,:len(carrierb1[i])] = carrierb1[i]
        new[7,i,:len(carrierb2[i])] = carrierb2[i]

        psi_all.append(psi)
        psib_all.append(psib)
        G_all.append(H)
        G_all_b.append(Hb)
        sat_pos_all.append(sat_pos)
        satb_pos_all.append(sat_pos)
        t_all.append(t_gps[i])



# # convert into np arrays
psi_all = np.array(psi_all)
psib_all = np.array(psib_all)
G_all = np.array(G_all)
G_all_b = np.array(G_all_b)
sat_pos_all = np.array(sat_pos_all)
satb_pos_all = np.array(satb_pos_all)
t_all = np.array(t_all)

### save them as np arrays
np.save('meas_gps.npy', psi_all)
np.save('meas_bei.npy', psib_all)
np.save('geom_gps.npy', G_all)
np.save('geom_bei.npy', G_all_b)
np.save('sat_pos_gps.npy', sat_pos_all)
np.save('sat_pos_bei.npy', satb_pos_all)
np.save('tgps.npy', t_all)
np.save('raw_meas.npy',new)

print ('code done')
print ('Done_all.. yeah!')



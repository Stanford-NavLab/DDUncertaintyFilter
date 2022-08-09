#from wsgiref.types import ErrorStream
import numpy as np
import pandas as pd
from numpy import genfromtxt
from new_utils import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## oh man.. need to run from main.py


#this needs to be fixed
base_ecef = np.array([-2414266.9197,5386768.9868, 2407460.0314]) #position of the base station in ECEF
x0_enu = np.array([832591.326, 820351.389, 20.2469])   
x0_lla = np.array([[22.322171,114.141188, 20.2386]]) #base station lla


#load the other files
psig = np.load('meas_gps.npy',allow_pickle=True)
psic = np.load('meas_bei.npy',allow_pickle=True)
Hg = np.load('geom_gps.npy',allow_pickle=True)
Hc = np.load('geom_bei.npy',allow_pickle=True)
sat_posg = np.load('sat_pos_gps.npy',allow_pickle=True)
sat_posc = np.load('sat_pos_bei.npy',allow_pickle=True)
time = np.load('tgps.npy',allow_pickle=True)  
raw = np.load('raw_meas.npy',allow_pickle=True)


print (len(time), len(psig), len(Hg), len(sat_posc), len(sat_posg))

ind = np.argsort(time)

psig = psig[ind]
psic = psic[ind]
Hg = Hg[ind]
Hc = Hc[ind]
sat_posg = sat_posg[ind]
sat_posc = sat_posc[ind]
tnew = time[ind]

raw = np.array([raw[:, i, :] for i in range(raw.shape[1]) if not (np.all(raw[:, i, :] == 0))])
raw = raw[ind]


# dd_code = (raw[0,0,:] - raw[0,1,:])[:, None] - (raw[0,0,:] - raw[0,1,:])[None, :]
dd_code = (raw[0,0,0] - raw[0,1,0]) - (raw[0,0,2] - raw[0,1,2])
print ('dd code',dd_code)
print ('psig',psig[0])

#print (psig, psic, Hg, Hc, sat_posc, sat_posg, tnew)


#print ('tnew',tnew)

gt = genfromtxt('medium_dataset/ground_truth.txt', delimiter = ',')
(x,y,z) = lla_to_ecef(gt[:,1], gt[:,2], gt[:,3]/1000)  #ground truth of rover ecef x,y,z

rover_ecef = np.array([x,y,z]).T

# out = np.array([ecef2enu(np.array([x[i], y[i], z[i]]), x0_lla[0,0], x0_lla[0,1],base_ecef) for i in range(len(rover_ecef))])

# print (out.shape)

# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(out[:, 0], out[:, 1], out[:, 2])
# plt.show()



# confirm about the date in the .obs file AND THE BASE STATION FILE

gt_times = gt[:,0]

f_gps = 1575.42 * 10 ** 6
f_bei = 1561.1 * 10 ** 6
c = 299792458
lda_gps = c / f_gps
lda_bei = c / f_bei


diff_code = []
diff_ca = []

## last half is code, first half is carrier
sat_ind = 0
res = np.zeros((8, 100))

for i in range(100):
   for j in range(len(tnew)):
      if abs(round(gt_times[i]) - round(tnew[j])) < 0.1:
         #print (abs(round(gt_times[i]) - round(tnew[j])))
         psi = psic[j]
         n = len (psi) // 2  # first half is la * carrier, second half is code
         dd_ca = psi[0:n] 
         dd_code = psi[n:]
         # print ('code', dd_code)
         # print ('ca', dd_ca)

         # do it for one satellite only, later extend to all satellites

         # BEIDOU distances
         a = np.linalg.norm(sat_posc[j][sat_ind] - rover_ecef[i])
         b = np.linalg.norm(sat_posc[j][sat_ind] - base_ecef)

         # term1_ca = (lda_gps/lda_bei) * (a - b)
         # term1_code = (a - b)


         # GPS distances
         # a = np.linalg.norm(sat_posg[j][sat_ind] - rover_ecef[i])
         # b = np.linalg.norm(sat_posg[j][sat_ind] - base_ecef)

         term1_ca = (a - b)
         term1_code = (a - b)

         # BEIDOU distance
         c = np.linalg.norm(sat_posc[j][-1] - rover_ecef[i])  # use the last gps satellite since that is used as reference
         d = np.linalg.norm(sat_posc[j][-1] - base_ecef)

         # GPS distances
         # c = np.linalg.norm(sat_posg[j][-1] - rover_ecef[i])  # use the last gps satellite since that is used as reference
         # d = np.linalg.norm(sat_posg[j][-1] - base_ecef)
         term2 = c - d

         truth_code = term1_code - term2
         truth_ca = term1_ca - term2
         # df_code = dd_code[sat_ind] - truth_code
         # df_ca = dd_ca[sat_ind] - truth_ca


         res[0,i] = raw[j,0,sat_ind] - c
         res[1,i] = raw[j,1,sat_ind] - d
         res[2,i] = raw[j,2,sat_ind] - a
         res[3,i] = raw[j,3,sat_ind] - b

         res[4,i] = lda_gps *raw[j,4,sat_ind] - c
         res[5,i] = lda_gps *raw[j,5,sat_ind] - d
         res[6,i] = lda_bei *raw[j,6,sat_ind] - a
         res[7,i] = lda_bei *raw[j,7,sat_ind] - b

         # REMOVE TRAILING ZEROS IN SATELLITE DIMENSION
         # dd_code = (raw[j,2,sat_ind] - raw[j,3,sat_ind]) - (raw[j,0,-1] - raw[j,1,-1])
         # dd_ca = lda_gps * ((raw[j,6,sat_ind] - raw[j,7,sat_ind]) - (raw[j,4,-1] - raw[j,5,-1]))

         last_non_zero_id = np.count_nonzero(raw[j,0,:]) - 1
         print("Should be same: ", len(sat_posg[j]), last_non_zero_id + 1)
         dd_code = (raw[j,2,sat_ind] - raw[j,3,sat_ind]) - (raw[j,2,last_non_zero_id] - raw[j,3,last_non_zero_id])
         
         # dd_ca = lda_gps * ((raw[j,4,sat_ind] - raw[j,5,sat_ind]) - (raw[j,4,last_non_zero_id] - raw[j,5,last_non_zero_id]))

         dd_ca = lda_bei * ((raw[j,6,sat_ind] - raw[j,7,sat_ind]) - (raw[j,6,last_non_zero_id] - raw[j,7,last_non_zero_id]))

         df_code = dd_code - truth_code
         df_ca = dd_ca - truth_ca


         #print (df_code, df_ca)

         #print (df_code)
         diff_code.append(df_code)
         diff_ca.append(df_ca)



# for i in range(8):
#    if i==0 or i==2 or i==4 or i==6: 
#       plt.figure()
#       plt.plot(res[i,:] - res[i+1,:])
#       plt.title(str(i))





plt.figure()
plt.plot(diff_code)
plt.title('truth - code phase DD')

plt.figure()
plt.plot(diff_ca)
plt.title('truth - carrier phase DD')
plt.show()





## used the last one from gps as the ref satellite


# #for i in range(len(time)):
#  #   k = np.array([x[i],y[i],z[i]])
#   #  out = ecef2enu(np.array([x[i],y[i],z[i]]), x0_lla[0,0], x0_lla[0,1],x0)
#    # (gt_enu[i,0], gt_enu[i,1], gt_enu[i,2]) = out[0], out[1], out[2]
 

# # true_amb = np.array(true)
# # est_amb = np.array(est)
# # df = pd.DataFrame(true_amb)
# # df.to_csv('True_high_0.01')
# # df = pd.DataFrame(est_amb)
# # df.to_csv('Est_high_0.01')

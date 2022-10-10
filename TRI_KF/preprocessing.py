# This file contains functions to process GNSS data from RINEX files
# Should contain in the future cleaner function (or class) to preprocess all data and data at one time step

import numpy as np
import xarray as xr
import georinex as gr
from utils import *
import os


def loadTrajectories(name_obs1, name_obs2, name_eph1, name_eph2):
    """
    Load rinex files into xarrays
    """
    print("Loading Ephemeris...")
    eph1 = gr.load(name_eph1)
    print("...Ephemeris 1 loaded")
    eph2 = gr.load(name_eph2)
    print("...Ephemeris 2 loaded")
    
    print("Loading Observations")
    obs1 = gr.load(name_obs1)
    print("...Observation 1 loaded")
    obs2 = gr.load(name_obs2)
    print("...Observation 2 loaded")
    
    return obs1, obs2, eph1, eph2

def loadSavedTrajectories():
    """
    Load rinex files into xarrays
    """
    return xr.open_dataset('data/traj1.nc'),xr.open_dataset('data/traj2.nc'),xr.open_dataset('data/eph.nc')

def removeDuplicateSatellites(obs):
    '''
    Remove duplicate satellites in xarray to fix a possible bug in georinex/xarray
    '''
    idxs=[]
    svs=[]
    for i in range(len(obs.sv.values)):
        if obs.sv.values[i] not in svs:
            idxs.append(i)
            svs.append(obs.sv.values[i])
    obs=obs.isel(sv=idxs)
    return obs



def constructMeasurements(traj1, traj2, date,sort_cn0 = False):
    """
    Construct necessary data at each time step
    Inputs:
    Trajectories as xarrays loaded from rinex files with georinex
    """
    t1 = set(traj1.time.values)
    t2 = set(traj2.time.values)
    svs = []
    svbs = []
    code1=[]
    code2=[]
    carrier1=[]
    carrier2=[]
    codeb1=[]
    codeb2=[]
    carrierb1=[]
    carrierb2=[]
    t_gps=[]
    for t in t1:
        for t_2 in t2:                
            k1 = timeInGPSWeek(t, date)
            k2 = timeInGPSWeek(t_2, date)  
            if abs(round(k1) - k2) < 1:
                t1_t = traj1.sel(time=t)
                t2_t = traj2.sel(time=t_2)
                
                code_t1 = t1_t['C1C'] 
                code_t2 = t2_t['C1C'] 

                codeb_t1 = t1_t['C2I'] 
                codeb_t2 = t2_t['C1I'] 

                sv1 = [t1_t.sv.values[i] for i in range(len(t1_t.sv.values)) if not np.isnan(code_t1[i])]
                sv2 = [t2_t.sv.values[i] for i in range(len(t2_t.sv.values)) if not np.isnan(code_t2[i])]
                sv = np.intersect1d(sv1,sv2)

                svb1 = [t1_t.sv.values[i] for i in range(len(codeb_t1)) if not np.isnan(codeb_t1[i])]
                svb2 = [t2_t.sv.values[i] for i in range(len(codeb_t2)) if not np.isnan(codeb_t2[i])]
                svb = np.intersect1d(svb1,svb2)

                #print ('gps sv', sv)
                print ('bei sv', svb)

                t1_t = t1_t.sel(sv=sv)
                t2_t = t2_t.sel(sv=sv)

                # t1b_t = t1_t.sel(sv=svb)
                # t2b_t = t2_t.sel(sv=svb)

                if sort_cn0:
                    order = np.argsort(t1_t['S1C'].values)
                    orderb = np.argsort(t1_t['S2I'].values)
                else:
                    order = np.arange(len(sv))
                    orderb = np.arange(len(svb))
                
                #print ('gps', order)
                print ('bei', orderb)

                code1.append(t1_t['C1C'].values[order] )
                carrier1.append(t1_t['L1C'].values[order])
                svs.append(sv[order])
                code2.append(t2_t['C1C'].values[order])
                carrier2.append(t2_t['L1C'].values[order]) 


                print (t1_t['C2I'].values)
                print (t2_t['C1I'].values)
                codeb1.append(t1_t['C2I'].values[orderb])
                carrierb1.append(t1_t['L2I'].values[orderb])
                svbs.append(svb[orderb])
                codeb2.append(t2_t['C1I'].values[orderb])
                carrierb2.append(t2_t['L1I'].values[orderb]) 
        t_gps.append(timeInGPSWeek(t, date))   
    
    return t_gps, svs, svbs, code1, code2, carrier1, carrier2, codeb1, codeb2, carrierb1, carrierb2


def constructMeasurements_bei(traj1, traj2, date,sort_cn0 = False):
    """
    Construct necessary data at each time step
    Inputs:
    Trajectories as xarrays loaded from rinex files with georinex
    """
    t1 = set(traj1.time.values)
    t2 = set(traj2.time.values)
    # t1 = sorted(list(t1))
    # t2 = sorted(list(t1))
    # print ('t1 bei', t1)
    # print ('t2 bei', t2)
    #print (ts)
    svs = []
    code1=[]
    code2=[]
    carrier1=[]
    carrier2=[]
    cnos=[]
    t_gps=[]
    for t in t1:
        for t_2 in t2:                
            k1 = timeInGPSWeek(t, date)
            k2 = timeInGPSWeek(t_2, date)  
            if abs(round(k1) - k2) < 1:
                # print (k1, k2)
                # print (abs(round(k1) - k2))  
                # print (t, t_2)
                # print (k1, k2)           
                t1_t = traj1.sel(time=t)
                if t1_t['C2I'].ndim > 1:
                    code_t1 = t1_t['C2I'][0] 
                else:
                    code_t1 = t1_t['C2I'] 

                t2_t = traj2.sel(time=t_2)
                if t2_t['C1I'].ndim > 1:
                    code_t2 = t2_t['C1I'][0]
                else:
                    code_t2 = t2_t['C1I'] 
                
                sv1 = [t1_t.sv.values[i] for i in range(len(t1_t.sv.values)) if not np.isnan(code_t1[i])]
                sv1_new = []
                for j in range(len(sv1)):
                    #print (sv1[j])
                    if sv1[j][0] == 'C':
                        sv1_new.append(sv1[j])
                #print ('sv1',sv1_new)

                sv2 = [t2_t.sv.values[i] for i in range(len(t2_t.sv.values)) if not np.isnan(code_t2[i])]
                sv2_new = []
                for j in range(len(sv2)):
                    #print (sv1[j])
                    if sv2[j][0] == 'C':
                        sv2_new.append(sv2[j])
                sv = np.intersect1d(sv1_new,sv2_new)
                if len(sv) is None:
                    print ('no intersection')
                try:
                    t1_t = t1_t.sel(sv=sv)
                    t2_t = t2_t.sel(sv=sv)
                except:
                    print('oops',sv)
                if sort_cn0:
                    if t1_t['C2I'].ndim > 1:
                        order = np.argsort(t1_t['S2I'][0].values)
                    else:
                        order = np.argsort(t1_t['S2I'].values)
                else:
                    if t1_t['C2I'].ndim > 1:
                        order = np.arange(len(t1_t['S2I'][0].values))
                    else:
                        order = np.arange(len(t1_t['S2I'].values))
                if t1_t['C2I'].ndim > 1:     
                    code1.append(t1_t['C2I'][0].values[order] )
                    carrier1.append(t1_t['L2I'][0].values[order])
                    cnos.append(t1_t['S2I'][0].values[order])
                    svs.append(sv[order])
                else:
                    code1.append(t1_t['C2I'].values[order] )
                    carrier1.append(t1_t['L2I'].values[order])
                    cnos.append(t1_t['S2I'].values[order])
                    svs.append(sv[order])
                if t2_t['C1I'].ndim > 1:
                    code2.append(t2_t['C1I'][0].values[order])
                    carrier2.append(t2_t['L1I'][0].values[order])
                else:
                    code2.append(t2_t['C1I'].values[order])
                    carrier2.append(t2_t['L1I'].values[order]) 
                break
        t_gps.append(timeInGPSWeek(t, date)) 
    return t_gps, svs, code1, code2, carrier1, carrier2, cnos, t_gps
                


def constructMeasurements_gps(traj1, traj2, date,sort_cn0 = False):
    """
    Construct necessary data at each time step
    Inputs:
    Trajectories as xarrays loaded from rinex files with georinex
    """
    t1 = set(traj1.time.values)
    t2 = set(traj2.time.values)


    # print ('a1',t1)
    # print ('a2', t2)

    #ts = sorted(list(t1.intersection(t2)))
    #print ('ts', ts)
    svs = []
    code1=[]
    code2=[]
    carrier1=[]
    carrier2=[]
    cnos=[]
    t_gps=[]
    for t in t1:
        for t_2 in t2:   
            #print (t, t2)             
            k1 = timeInGPSWeek(t, date)
            k2 = timeInGPSWeek(t_2, date)  
            if abs(round(k1) - round(k2)) < 1:       
                t1_t = traj1.sel(time=t)
                if t1_t['C1C'].ndim > 1:
                    code_t1 = t1_t['C1C'][0] 
                else:
                    code_t1 = t1_t['C1C'] 

                t2_t = traj2.sel(time=t_2)
                if t2_t['C1C'].ndim > 1:
                    code_t2 = t2_t['C1C'][0]
                else:
                    code_t2 = t2_t['C1C'] 
                
                sv1 = [t1_t.sv.values[i] for i in range(len(t1_t.sv.values)) if not np.isnan(code_t1[i])]
                sv1_new = []
                for j in range(len(sv1)):
                    #print (sv1[j])
                    if sv1[j][0] == 'G':
                        sv1_new.append(sv1[j])
                #print ('sv1',sv1_new)

                sv2 = [t2_t.sv.values[i] for i in range(len(t2_t.sv.values)) if not np.isnan(code_t2[i])]
                sv2_new = []
                for j in range(len(sv2)):
                    #print (sv1[j])
                    if sv2[j][0] == 'G':
                        sv2_new.append(sv2[j])
                #print ('sv2',sv2_new)
                #print ('sv2', sv2)
                #print (len(code_t2))

                sv = np.intersect1d(sv1_new,sv2_new)
                if len(sv) is None:
                    print ('no intersection')
                try:
                    t1_t = t1_t.sel(sv=sv)
                    t2_t = t2_t.sel(sv=sv)
                except:
                    print('oops',sv)
                if sort_cn0:
                    if t1_t['C1C'].ndim > 1:
                        order = np.argsort(t1_t['S1C'][0].values)
                    else:
                        order = np.argsort(t1_t['S1C'].values)
                else:
                    if t1_t['C1C'].ndim > 1:
                        order = np.arange(len(t1_t['S1C'][0].values))
                    else:
                        order = np.arange(len(t1_t['S1C'].values))
                if t1_t['C1C'].ndim > 1:     
                    code1.append(t1_t['C1C'][0].values[order] )
                    carrier1.append(t1_t['L1C'][0].values[order])
                    cnos.append(t1_t['S1C'][0].values[order])
                    svs.append(sv[order])
                else:
                    code1.append(t1_t['C1C'].values[order] )
                    carrier1.append(t1_t['L1C'].values[order])
                    cnos.append(t1_t['S1C'].values[order])
                    svs.append(sv[order])
                if t2_t['C1C'].ndim > 1:
                    code2.append(t2_t['C1C'][0].values[order])
                    carrier2.append(t2_t['L1C'][0].values[order])
                else:
                    code2.append(t2_t['C1C'].values[order])
                    carrier2.append(t2_t['L1C'].values[order]) 
                break
        t_gps.append(timeInGPSWeek(t, date)) 
    return t_gps, svs, code1, code2, carrier1, carrier2, cnos, t_gps

def prepareData(t, svs, code1, code2, carrier1, carrier2, eph, 
                plane=False,ref=0,x0=x0,f=1575.42*10**6,
                phase_error=0.025,sigma_code=None,sigma_phase=None):
    """
    Generate psi, H, A and sigma for the optimization problem.
    Works both in 2D and 3D with the plane variable
    For 2D need to add computation of H in ENU
    Inputs:
    t: current date in seconds in the GPS week
    svs: list of satellite in views by both recieevrs at time t
    code1(2): list for all code measurments of receiver 1 (2) in same orders as svs
    carrier1(2): same for carrier phase
    eph: xarray with rinex navigation file loaded
    plane: whether to work in local ENU frame (not yet implemented)
    ref: reference for double difference computation
    date: day (without hours/minute/seconds) of the experiment
    x0: position for geometry matrix computation
    f: measurement frequency
    phase_error: assumed error in meter in carrier phase measurements for default noise estimation
    sigma_code,sigma_phase: can be used to specify noise standard deviations
    """
    c=299792458
    lda = c/f
    n=len(svs) -1
    ft = computeFlightTimes(code1, svs, eph,t)
    (H, sat_pos)= computeGeometry(eph, t, ft, svs, x0, ref, plane)
    
    # call regularly on gps
    psi = computeDD(code1, code2, carrier1, carrier2, lda, ref)
    
    # call with beidou but using gps as ref

    
    # A = np.zeros((2*n,n))
    # A[:n]=lda*np.eye(n)
    #sigma= computeSigma2(n,sigma_code,sigma_phase,f,phase_error,ref)
    return psi, H, sat_pos


def prepareData_mixed(t, svs, svbs, code1, code2, codeb1, codeb2, carrier1, carrier2, carrierb1, carrierb2,gps_eph, bei_eph, lda_gps, lda_bei,plane=True, ref= 0, x0=x0):
    n = len(svs) + len(svbs) - 1
    ft = computeFlightTimes(code1, svs, gps_eph,t)
    ftb = computeFlightTimes(codeb1, svbs, bei_eph,t)

    # print ('ftb', ftb)
    # print ('svbs', svbs)

    (H, Hb, sat_pos, satb_pos)= computeGeometry_mixed(gps_eph, bei_eph, t, ft, ftb, svs, svbs, x0, ref, plane)  # ask Shubh
    psi = computeDD(code1, code2, carrier1, carrier2, lda_gps, ref)
    psib = computeDD_mixed(codeb1, codeb2, carrierb1, carrierb2, lda_bei, code1, code2, carrier1, carrier2, lda_gps,ref)  # ask Shubh.. confused here ..revisit

    return psi, psib,H, Hb, sat_pos, satb_pos
 





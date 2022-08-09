import numpy as np
import pandas as pd
from numpy import genfromtxt


def timeInGPSWeek(t, date):
    '''
    Compute time in the GPS week
    '''
    dt = date-np.datetime64('1980-01-06')
    dt = dt.astype('float64')
    nWeeks = np.floor(dt/7)
    
    return (t-np.datetime64('1980-01-06')).astype('float64')*1e-9-nWeeks*7*24*60*60


def lla_to_ecef(lat,lon,alt):


    """This function takes in WGS84 latitude, longitude, and altitude 
       from LLA coordinate system and converts to ECEF coordinate system
    Args:
        lat [pandas.core.series.Series]: WGS84 Latitude [degrees]
                                         Example: df.latitude
        lon [pandas.core.series.Series]: WGS84 Longitude [degrees]
                                         Example: df.longitude
        alt [pandas.core.series.Series]: WGS84 Altitude [kilometers]
                                         Example: df.altitude
    Returns:
       X [pandas.core.series.Series]: ECEF x-coordinate [meters]
       Y [pandas.core.series.Series]: ECEF y-coordinate [meters]
       Z [pandas.core.series.Series]: ECEF z-coordinate [meters]
    """    
    # convert lat and long from degrees to radians because numpy expects 
    # radians for trig functions
    deg_2_rads = np.pi/180
    lat = deg_2_rads*lat
    lon = deg_2_rads*lon    
    
    # convert altitude from kilometers to meters
    alt = 1000*alt    
    
    # convert LLA to ECEF with the following equations
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    sin_lat = np.sin(lat)

    A = 6378137
    B = 6356752.31424518
    H = alt
    E1 = np.sqrt((A**2-B**2)/A**2)
    E2 = E1**2
    N = A/np.sqrt(1-E2*(sin_lat**2))

    X = (N+H)*cos_lat*cos_lon
    Y = (N+H)*cos_lat*np.sin(lon)
    Z = (N*(1-E2)+H)*sin_lat
    
    return X,Y,Z



def ecef2enu(x, lat0, lon0 , x0):
    
    phi = np.radians(lat0)
    lda = np.radians(lon0)

    sl = np.sin(lda)
    cl = np.cos(lda)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    x = x- x0

    x_enu = np.zeros(3)
    x_enu[0] = -sl * x[0] + cl * x[1]
    x_enu[1] = -cl * sp * x[0] - sl * sp * x[1] + cp * x[2]
    x_enu[2] = cl * cp * x[0] + sl * cp * x[1] + sp * x[2]

    return x_enu
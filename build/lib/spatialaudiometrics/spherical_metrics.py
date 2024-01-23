'''
spherical_metrics.py. Functions to calculate spherical metrics

Copyright (C) 2024  Katarina C. Poole

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
'''
from math import degrees, acos
import numpy as np

def polar2cartesian(az,el,dist):
    '''
    Converts polar coordinates (azimuth and elevation) to cartesian
    '''
    az = np.deg2rad(az)
    el = np.deg2rad(el)
    x = dist * np.cos(az) * np.cos(el)
    y = dist * np.sin(az) * np.cos(el)
    z = dist * np.sin(el)
    return x,y,z

def great_circle_error(az1,el1,az2,el2):
    '''
    Calculate the great circle arror between two azimuth and elevation locations
    '''
    x1,y1,z1 = polar2cartesian(az1,el1,1)
    x2,y2,z2 = polar2cartesian(az2,el2,1)
    coords1 = [x1,y1,z1]
    coords2 = [x2,y2,z2]
    dot_prod = np.dot(coords1,coords2)
    angle = degrees(acos(dot_prod))
    return angle

def spherical2interaural(az,el):
    '''
    Converts spherical (azimuth and elevation) to interaural cordinates (lateral and polar)
    lat	lateral angle in deg, [-90 deg, +90 deg]
    pol	polar angle in deg, [-90 deg, 270 deg]
    Currently doesn't take array 
    (would need to fix the ifs statements to work with np.where but would also need to get it to work with single numbers)
    
    Modified by Katarina C. Poole from the AMT toolbox sph2horpolar.m (23/01/2024)
    Url: http://amtoolbox.org/amt-1.5.0/doc/common/sph2horpolar.php
    Original author: Peter L. Sondergaard

    '''
    # Firstly need to make sure the azi angles are bound between 0 and 360
    az      = wrap_angle(az,0,360)
    el      = wrap_angle(el,0,360)
    razi    = np.deg2rad(az)
    rele    = np.deg2rad(el)
    # Calculate lateral angle
    rlat    = np.arcsin(np.sin(razi)*np.cos(rele))
    lat     = np.rad2deg(rlat)
    # Then calculate polar
    if np.round(np.cos(rlat),15)==0:
        rpol = 0
    else:
        rpol = np.arcsin(np.round(np.sin(rele),15)/np.round(np.cos(rlat),15)) # Some times get wierdness in precision so do a bit of rounding
    pol      = np.rad2deg(rpol)

    # Make some corrections on the polar angle
    if ((az > 90) and (az < 270) and ((el < 90) or (el > 270))):
        pol = 180 - pol
    if not((az > 90) and (az < 270)) and (el > 90) and (el < 270):
        pol = 180 - pol

    # Then do some rounding
    lat = np.round(lat,5)
    pol = np.round(pol,5)
    return lat, pol

def wrap_angle(angle,low = -180,high = 180):
    '''
    Wraps the angle between a specific range.
    Default is -180 and 180 to get it to work well with 
    the apply function in pandas
    '''
    wrap_range = high-low
    angle =  angle%wrap_range

    if angle > high:
        angle = angle - wrap_range
    elif angle < low:
        angle = angle + wrap_range
    return angle

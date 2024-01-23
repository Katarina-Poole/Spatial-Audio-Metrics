'''
spherical_metrics.py. Functions to calculate spherical metrics

'''
from math import degrees, acos
import numpy as np

def polar2cartesian(az:float, el:float, dist:float):
    '''
    Converts polar coordinates (azimuth and elevation) to cartesian

    :param az: Azimuth coordinate
    :param el: Elevation coordinate
    :param dist: Distance coordinate
    '''
    az = np.deg2rad(az)
    el = np.deg2rad(el)
    x = dist * np.cos(az) * np.cos(el)
    y = dist * np.sin(az) * np.cos(el)
    z = dist * np.sin(el)
    return x,y,z

def great_circle_error(az1:float, el1:float, az2:float, el2:float):
    '''
    Calculate the great circle arror between two azimuth and elevation locations
    
    :param az1: First azimuth coordinate
    :param el1: First elevation coordinate
    :param az2: Second azimuth coordinate
    :param el2: Second elevation coordinate
    '''
    x1,y1,z1 = polar2cartesian(az1,el1,1)
    x2,y2,z2 = polar2cartesian(az2,el2,1)
    coords1 = [x1,y1,z1]
    coords2 = [x2,y2,z2]
    dot_prod = np.dot(coords1,coords2)
    angle = degrees(acos(dot_prod))
    return angle

def spherical2interaural(az:float, el:float):
    '''
    Converts spherical (azimuth and elevation) to interaural cordinates (lateral and polar)

    | lat	lateral angle in deg, [-90 deg, +90 deg]
    | pol	polar angle in deg, [-90 deg, 270 deg]
    | Currently doesn't take array (would need to fix the ifs statements to work with np.where but would also need to get it to work with single numbers)
    | Modified by Katarina C. Poole from the AMT toolbox sph2horpolar.m (23/01/2024)
    | Url: http://amtoolbox.org/amt-1.5.0/doc/common/sph2horpolar.php
    | Original author: Peter L. Sondergaard

    :param az: Azimuth coordinate
    :param el: Elevation coordiate
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

def wrap_angle(angle:float,low = -180,high = 180):
    '''
    Wraps the angle between a specific range.

    :param angle: Angle in degrees to wrap
    :param low: Minimum angle to wrap to (default is -180 degrees)
    :param high: Maximum angle to wrap to (default is 180 degrees)
    '''

    wrap_range = high-low
    angle =  angle%wrap_range

    if angle > high:
        angle = angle - wrap_range
    elif angle < low:
        angle = angle + wrap_range
    return angle

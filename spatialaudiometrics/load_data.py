'''
Load data module
'''
import os
import sys
from importlib import resources
from pysofaconventions import SOFAFile
import numpy as np
import pandas as pd
from spatialaudiometrics import angular_metrics as am
from spatialaudiometrics import localisation_metrics as lm

class HRTF:
    '''
    A general structure for loading all info needed for an HRTF since we can't edit the sofa files with pysofaconventions easily
    '''
    def __init__(self,sofa_path):
        '''
        Loads in the data from the sofa path into the custom class

        :param sofa: sofa object from SOFAFile(sofa_path,'r')
        '''
        # Firstly check if each path exists
        assert os.path.isfile(sofa_path) is True, 'Sofa file does not exist'
        
        sofa            = SOFAFile(sofa_path,'r')
        self.sofa_path  = sofa_path
        self.locs       = np.round(sofa.getVariableValue('SourcePosition').data,2) # Round to avoid any bizarre precision errors
        self.hrir       = sofa.getDataIR().data
        self.fs         = sofa.getSamplingRate().data[0]

def load_example_behavioural_data():
    '''
    Loads in example data from a spatial audio experiment (example_data_1.csv)
    '''
    with resources.path("spatialaudiometrics","example_data_1.csv") as df:
        return pd.read_csv(df)
    
def load_example_sofa_files():
    '''
    Loads in two example sofa files from the SONICOM dataset and returns custom hrtf objects
    '''
    with resources.path("spatialaudiometrics","example_sofa_1.sofa") as sofa:
        hrtf1 = HRTF(sofa)
    with resources.path("spatialaudiometrics","example_sofa_2.sofa") as sofa:
        hrtf2 = HRTF(sofa)
    return hrtf1,hrtf2


def match_hrtf_locations(hrtf1,hrtf2):
    '''
    Finds where the locations of hrtf 2 match hrtf 1 and then reorders the HRIRs such that they match. Currently the distance dimension is ignored and only azimuth and elevation are taken into account
    
    :param hrtf1: custom hrtf object 
    :param hrtf2: custom hrtf in which the hrirs will be reordered to match the locations of sofa1
    '''
    loc2_idx = list()
    for i,loc in enumerate(hrtf1.locs):
        idx = np.where((hrtf2.locs[:,0] == loc[0]) & (hrtf2.locs[:,1] == loc[1]))[0]
        if len(idx) > 0:
            loc2_idx.append(idx[0])
        elif len(idx) == 0:
            print('Could not find a match for az: ' + str(loc[0]) + ' and el: ' + str(loc[1]))
        else:
            print('Multiple matches, picking the first match for location az: ' + str(loc[0]) + ' and el: ' + str(loc[1]))
            loc2_idx.append(idx[0])

    if len(loc2_idx) == len(hrtf1.locs):
        print('Successfully matched all locations to hrtf 1')
    else:
        sys.exit('Error: Was not able to match all the locations in hrtf 1 with hrtf 2')
    hrtf2.hrir          = hrtf2.hrir[loc2_idx,:,:]
    hrtf2.locs          = hrtf2.locs[loc2_idx,:]
    return hrtf1, hrtf2

def preprocess_behavioural_data(df:pd.DataFrame):
    '''
    Preprocesses the data to make sure we generate the information needed to calculate the metrics
    
    | 1. Wraps azimuth/elevation angles between -180 and 180 (negative = counterclockwise/down)
    | 2. Adds in the spherical coordinate system (lateral and polar angles)
    | 3. Calculate errors

    :param df: Pandas dataframe (every row should be a trial and should include the columns 'ele_target', 'ele_response', 'azi_target', 'azi_response')
    :return df: Returns the dataframe with added columns and new coordinates under the assumption that -90 = left, 90 = right in azi, and -90 = down and 90 = top in ele
    '''

    # Wrap azi and ele angles between -180 and 180
    df['azi_target']                = df.azi_target.apply(am.wrap_angle)
    df['azi_response']              = df.azi_response.apply(am.wrap_angle)
    df['ele_target']                = df.ele_target.apply(am.wrap_angle)
    df['ele_response']              = df.ele_response.apply(am.wrap_angle)

    # Initialise the new columns to be added
    df['lat_target']                = np.nan
    df['lat_response']              = np.nan
    df['pol_target']                = np.nan
    df['pol_response']              = np.nan
    df['great_circle_error']        = np.nan
    df['confusion_classification']  = ''

    for i,row in df.iterrows():
        # Calculate interaural corrdinates
        lat_target,pol_target       = am.spherical2interaural(df.at[i,'azi_target'],df.at[i,'ele_target'])
        lat_response,pol_response   = am.spherical2interaural(df.at[i,'azi_response'],df.at[i,'ele_response'])
        df.at[i,'lat_target']       = lat_target
        df.at[i,'pol_target']       = pol_target
        df.at[i,'lat_response']     = lat_response
        df.at[i,'pol_response']     = pol_response

        # Calculate great circle error
        df.at[i,'great_circle_error'] = am.great_circle_error(df.at[i,'azi_target'],df.at[i,'ele_target'],
                                                              df.at[i,'azi_response'],df.at[i,'ele_response'])
        
        # Classify confusions
        df.at[i,'confusion_classification'] = lm.classify_confusion(df.iloc[i])

    # Calculate weighting
    df['polar_weight']           = lm.polar_error_weight(df)

    # Calculate errors
    df['signed_lateral_error']   = df.lat_response - df.lat_target
    df['signed_polar_error']     = (df.pol_response - df.pol_target)*df.polar_weight
    
    # Wrap signed errors between 180 and -180
    df['signed_lateral_error']   = df.signed_lateral_error.apply(am.wrap_angle)
    df['signed_polar_error']     = df.signed_polar_error.apply(am.wrap_angle)
    
    # Calculate the unsigned version
    df['unsigned_lateral_error'] = abs(df.signed_lateral_error)
    df['unsigned_polar_error']   = abs(df.signed_polar_error)

    return df
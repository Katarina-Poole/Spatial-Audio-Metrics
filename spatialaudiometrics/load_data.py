'''
load_example_data.py. Functions that loads in example data
'''
from importlib import resources
import numpy as np
import pandas as pd
from spatialaudiometrics import spherical_metrics as sm
from spatialaudiometrics import localisation_metrics as lm

def load_example_behavioural_data():
    '''
    Loads in example data from a spatial audio experiment (example_data_1.csv)
    '''
    with resources.path("spatialaudiometrics","example_data_1.csv") as df:
        return pd.read_csv(df)
    print(df)

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
    df['azi_target']                = df.azi_target.apply(sm.wrap_angle)
    df['azi_response']              = df.azi_response.apply(sm.wrap_angle)
    df['ele_target']                = df.ele_target.apply(sm.wrap_angle)
    df['ele_response']              = df.ele_response.apply(sm.wrap_angle)

    # Initialise the new columns to be added
    df['lat_target']                = np.nan
    df['lat_response']              = np.nan
    df['pol_target']                = np.nan
    df['pol_response']              = np.nan
    df['great_circle_error']        = np.nan
    df['confusion_classification']  = ''

    for i,row in df.iterrows():
        # Calculate interaural corrdinates
        lat_target,pol_target       = sm.spherical2interaural(df.at[i,'azi_target'],df.at[i,'ele_target'])
        lat_response,pol_response   = sm.spherical2interaural(df.at[i,'azi_response'],df.at[i,'ele_response'])
        df.at[i,'lat_target']       = lat_target
        df.at[i,'pol_target']       = pol_target
        df.at[i,'lat_response']     = lat_response
        df.at[i,'pol_response']     = pol_response

        # Calculate great circle error
        df.at[i,'great_circle_error'] = sm.great_circle_error(df.at[i,'azi_target'],df.at[i,'ele_target'],
                                                              df.at[i,'azi_response'],df.at[i,'ele_response'])
        
        # Classify confusions
        df.at[i,'confusion_classification'] = lm.classify_confusion(df.iloc[i])

    # Calculate weighting
    df['polar_weight']           = lm.polar_error_weight(df)

    # Calculate errors
    df['signed_lateral_error']   = df.lat_response - df.lat_target
    df['signed_polar_error']     = (df.pol_response - df.pol_target)*df.polar_weight
    # Wrap signed errors between 180 and -180
    df['signed_lateral_error']   = df.signed_lateral_error.apply(sm.wrap_angle)
    df['signed_polar_error']     = df.signed_polar_error.apply(sm.wrap_angle)
    # Calculate the unsigned version
    df['unsigned_lateral_error'] = abs(df.signed_lateral_error)
    df['unsigned_polar_error']   = abs(df.signed_polar_error)

    return df

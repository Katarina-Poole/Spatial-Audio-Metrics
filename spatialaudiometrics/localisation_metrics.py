'''
localisation_metrics.py. Functions that calculate perceptual metrics for localisation experiments

Copyright (C) 2024  Katarina C. Poole

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
import numpy as np
import pandas as pd
import scipy.stats as sts
import spherical_metrics as sm

def calculate_localisation_error(df,*grouping_vars):
    '''
    Calculates localisation precision and accuracy (lat and pol) like in AMT (currently only in the interaural domain for now)
    and the middle brooks quadrant error and confusion classification percentages
    :param df: data frame thats been preprocessed by load_behav_data.preprocess_localisation_data
    :grouping_vars: columns that you want to group the table when calculating the mean (e.g. 'subjectID','HRTFidx')
    '''
    grouping_list = list()
    for i,group_name in enumerate(grouping_vars):
        grouping_list.append(group_name)
    # Firstly need to split by subject since they average across 
    outdf = pd.DataFrame()
    grouped = df.groupby(grouping_list)
    for g,group in grouped:
        # Lateral accuracy and precision
        lat_accuracy        = np.mean(group.lat_response-group.lat_target)
        lat_abs_accuracy    = np.mean(np.abs(group.lat_response-group.lat_target))

        lat_precision       = np.std(group.lat_response-group.lat_target) # Dont need circular as not bound
        lat_abs_precision   = np.std(np.abs(group.lat_response-group.lat_target)) # Dont need circular as not bound

        # For the circular stats modules don't need to wrap errors which is nice! Also make sure the polar errors are weighted due to pole compression
        pol_accuracy        = sts.circmean((group.pol_response-group.pol_target)*group.polar_weight,high = 270, low = -90)
        pol_abs_accuracy    = sts.circmean(np.abs(group.pol_response-group.pol_target)*group.polar_weight,high = 270, low = -90)

        pol_precision       = sts.circstd((group.pol_response-group.pol_target)*group.polar_weight,high = 270, low = -90)
        pol_abs_precision   = sts.circstd(np.abs(group.pol_response-group.pol_target)*group.polar_weight,high = 270, low = -90)

        # Calculate middle brooks quadrant error
        quadrant_error,confusions,responses_within_lateral_range = calculate_quadrant_error(group)

        # Calculate the mean number of confusions as defined by Quinot
        precision_confusion     = (len(group.loc[(group.confusion_classification == 'precision')])/len(group))*100
        front_back_confusion    = (len(group.loc[(group.confusion_classification == 'front-back')])/len(group))*100
        in_cone_confusion       = (len(group.loc[(group.confusion_classification == 'in-cone')])/len(group))*100
        off_cone_confusion      = (len(group.loc[(group.confusion_classification == 'off-cone')])/len(group))*100

        temp = pd.DataFrame([[lat_accuracy,lat_precision,pol_accuracy,pol_precision,
                              lat_abs_accuracy,lat_abs_precision,pol_abs_accuracy,pol_abs_precision,
                              quadrant_error,confusions,responses_within_lateral_range,
                              precision_confusion,front_back_confusion,in_cone_confusion,off_cone_confusion]],
                            columns = ['lat_accuracy','lat_precision','pol_accuracy','pol_precision',
                              'lat_abs_accuracy','lat_abs_precision','pol_abs_accuracy','pol_abs_precision',
                              'quadrant_error','confusions','responses_within_lateral_range',
                              'precision_confusion','front_back_confusion','in_cone_confusion','off_cone_confusion'])
        for group_name in grouping_list:
            temp[group_name] = group[group_name].unique()[0] # Will always take the first

        outdf = pd.concat([outdf,temp])
    return outdf

def calculate_quadrant_error(df):
    '''
    Calculates the middlebrooks QE
    '''
    # Only want to look at when the response was responding in the front or back 60 degree cone
    curr_df         = df.loc[(np.abs(df.lat_response) <= 30)]
    #curr_df         = df.loc[(abs(df.lat_target) <= 30)]

    polar_error     = np.abs((curr_df.pol_response - curr_df.pol_target).apply(sm.wrap_angle))
    polar_error_idx = polar_error <= 90
    confusions      = sum(polar_error_idx)
    responses_within_lateral_range = len(curr_df)
    querr           = 100 - (confusions/responses_within_lateral_range)*100

    return querr, confusions,responses_within_lateral_range

def polar_error_weight(df):
    '''
    Calculates the weight for the polar error 
    '''
    w = 0.5*np.cos(2*np.deg2rad(df.lat_target)) + 0.5
    return w

def classify_confusion(row):
    '''
    Classifies whether the response is a:
    - Precision error (within 45degrees around target)
    - Front-back error (within 45 degrees of the opposite hemifield of the target)
    - In-cone error
    - Off-cone error
    - Combined - need to do this still!

    :param row: One row of the dataframe
    '''
    error = sm.great_circle_error(row.azi_target,row.ele_target,row.azi_response,row.ele_response)
    if error <= 45:
        classification = 'precision'
    else: # Check if its front back so get the opposite side azimuth angle
        error = sm.great_circle_error(sm.wrap_angle(-(180+row.azi_target)),row.ele_target,row.azi_response,row.ele_response) 
        if error <= 45:
            classification = 'front-back'
        elif np.abs(row.lat_response - row.lat_target) <= 45:
            classification = 'in-cone'
        else:
            classification = 'off-cone'

    return classification

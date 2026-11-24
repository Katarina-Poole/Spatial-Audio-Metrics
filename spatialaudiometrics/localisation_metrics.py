'''
localisation_metrics.py. Functions that calculate perceptual metrics for localisation experiments
'''
import numpy as np
import pandas as pd
import scipy.stats as sts
from spatialaudiometrics import angular_metrics as am

def calculate_localisation_error(df: pd.DataFrame,*grouping_vars: str, weighted_method: str = 'cos'):
    '''
    Calculates localisation precision and accuracy (lat and pol) like in AMT toolbox (currently only in the interaural domain for now)
    and the quadrant error and confusion classification percentages

    The polar errors and the quadrant error are weighted with polar_error_weight(group, weighted_method), because of pole compression:
    'cos' (default), 'cos2', 'split' (only targets within 30 degrees of the median plane, as the middlebrooks quadrant error) or 'none'.

    :param df: data frame thats been preprocessed by load_behav_data.preprocess_localisation_data
    :param grouping_vars: columns that you want to group the table when calculating the mean (e.g. 'subjectID','HRTFidx')
    :param weighted_method: weight of the polar errors and the quadrant error, see polar_error_weight
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

        # Polar accuracy and precision only for the responses without quadrant error (|polar error| <= 90 degrees,
        # before weighting), as middlebrooks: with front-back confusions the polar errors are bimodal,
        # and their mean and spread do not describe that distribution.
        # The error is wrapped, and weighted due to pole compression.
        w,used              = polar_error_weight(group, method=weighted_method)
        error               = (group.pol_response-group.pol_target).apply(am.wrap_angle)
        local               = used & (np.abs(error) <= 90)
        pol_error           = error[local]*w[local]
        pol_accuracy        = sts.circmean(pol_error,high = 270, low = -90)
        pol_abs_accuracy    = np.mean(np.abs(pol_error))

        pol_precision       = sts.circstd(pol_error,high = 270, low = -90)
        pol_abs_precision   = np.std(np.abs(pol_error))

        # Calculate the quadrant error
        quadrant_error,confusions,responses_within_lateral_range = calculate_quadrant_error(group, weighted_method)

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

def calculate_quadrant_error(df: pd.DataFrame, weighted_method: str = 'cos'):
    '''
    Calculates the quadrant error: the percentage of responses with an absolute polar error larger than 90 degrees

    All responses are used and each one is weighted with polar_error_weight(df, weighted_method), so that lateral targets, where the polar error is compressed, count less.
    The denominator is then the same for every listener and model. With weighted_method='split' it is the middlebrooks quadrant error like that in the AMT toolbox,
    which only uses targets within 30 degrees of the median plane. In our data (Measured HRTFs, 19 listeners x 2 sessions) the default 'cos' correlates at r = 0.84 with it.

    :param df: Pandas data frame with the columns: lat_target, lat_response, pol_target, pol_response which has the polar and lateral coordinates for behavioural responses and auditory targets
    :param weighted_method: weight of each response, see polar_error_weight: 'cos' (default), 'cos2', 'split' (middlebrooks) or 'none'
    '''
    w,used          = polar_error_weight(df, method=weighted_method)
    curr_df         = df.loc[used]
    w               = w.loc[used]

    polar_error     = np.abs((curr_df.pol_response - curr_df.pol_target).apply(am.wrap_angle))
    polar_error_idx = polar_error <= 90
    confusions      = sum(polar_error_idx)
    responses_within_lateral_range = len(curr_df)
    querr           = 100 - (np.sum(w*polar_error_idx)/np.sum(w))*100

    return querr, confusions,responses_within_lateral_range

def polar_error_weight(df: pd.DataFrame, method: str = 'cos'):
    '''
    Calculates the weight for the polar error such that lateral targets that are at -90 and 90 degrees have a low weight due to polar compression
    and which responses are used (boolean mask): all of them, except for method 'split', where the targets outside 30 degrees are left out, not counted as zero error.

    :param df: Pandas dataframe with the column lat_target (lateral target)
    :param method: 'cos' (default): cos(lat), the arc of a polar angle shrinks with cos(lat) away from the median plane,
        as the radius of the sagittal plane (Middlebrooks 1999, 10.1121/1.427147).
        'cos2': 0.5*cos(2*lat) + 0.5 = cos^2(lat) as in Majdak et al. (2010) 10.3758/APP.72.2.454 (default before PR #8).
        'split': constant weight (1) for the targets within 30 degrees of the median plane (|lat_target| <= 30), the others are not used, as the middlebrooks quadrant error (Middlebrooks 1999).
        'none': no weighting.
    :return: weight, used (both with the index of df)
    '''
    used = pd.Series(True, index=df.index)
    lat  = np.deg2rad(df.lat_target)
    if method == 'cos':
        w = np.cos(lat)
    elif method == 'cos2':
        w = 0.5*np.cos(2*lat) + 0.5
    elif method == 'split':
        used = np.abs(df.lat_target) <= 30
        w    = pd.Series(1.0, index=df.index)
    elif method == 'none':
        w = pd.Series(1.0, index=df.index)
    else:
        raise ValueError(f"Unknown method '{method}', use 'cos', 'cos2', 'split' or 'none'")
    return w, used

def classify_confusion(row,cone_size_degrees = 45):
    '''
    Classifys perceptual confusions as per Poirier-Quinot et al. (2022): 10.5772/intechopen.104931

    | Classifies whether the response is a:
    | - Precision error (within 45 degrees around target as default)
    | - Front-back error (within 45 degrees as default of the opposite hemifield of the target)
    | - In-cone error (errors made within the lateral cone, i.e. responses that failed at elevation judgements but were good in azimuth judgements)
    | - Off-cone error (errors that failed in azimuth judgements, outside of the cone)

    :param row: One row of a pandas dataframe with the columns 'azi_target, ele_target, azi_response, ele_response, lat_response, lat_target, pol_response and pol_target
    '''
    error = am.great_circle_error(row.azi_target,row.ele_target,row.azi_response,row.ele_response)
    if error <= cone_size_degrees:
        classification = 'precision'
    else: # Check if its front back so get the opposite side azimuth angle
        error = am.great_circle_error(am.wrap_angle(-(180+row.azi_target)),row.ele_target,row.azi_response,row.ele_response)
        if error <= cone_size_degrees:
            classification = 'front-back'
        elif np.abs(row.lat_response - row.lat_target) <= cone_size_degrees:
            classification = 'in-cone'
        else:
            classification = 'off-cone'

    return classification

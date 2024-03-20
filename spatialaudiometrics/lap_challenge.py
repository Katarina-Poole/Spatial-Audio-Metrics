'''
Functions to specifically deal with the LAP challenge
'''
import pandas as pd
from spatialaudiometrics import load_data as ld
from spatialaudiometrics import hrtf_metrics as hf

class Parameters:
    '''
    Parameters derived from a dataset
    '''
    itd_threshold = 32
    ild_threshold = 4
    lsd_threshold = 7
    
def calculate_lap_challenge_metrics(original_hrtf_path,upsampled_hrtf_path):
    '''
    Function that calculates all the metrics for the lab challenge
    '''
    # Load in the sofa files
    hrtf1 = ld.HRTF(original_hrtf_path)
    hrtf2 = ld.HRTF(upsampled_hrtf_path)
    
    # Match the locations
    hrtf1,hrtf2 = ld.match_hrtf_locations(hrtf1,hrtf2)

    # Calculate the metrics
    itd_diff = hf.calculate_itd_difference(hrtf1,hrtf2)
    ild_diff = hf.calculate_ild_difference(hrtf1,hrtf2)
    lsd,lsd_mat = hf.calculate_lsd_across_locations(hrtf1.hrir,hrtf2.hrir,hrtf1.fs)
    
    metric_names    = ['ITD difference (µs)','ILD difference (dB)','LSD (dB)']
    metrics         = [itd_diff,ild_diff,lsd]
    thresholds      = [Parameters.itd_threshold,Parameters.ild_threshold,Parameters.lsd_threshold]
    threshold_bool  = [itd_diff<Parameters.itd_threshold,
                       ild_diff<Parameters.ild_threshold,
                       lsd<Parameters.lsd_threshold]
    
    # dictionary of lists 
    dict = {'Metric name': metric_names,'Calculated value': metrics, 'Threshold value':thresholds,'Below threshold?': threshold_bool} 
    df = pd.DataFrame(dict)
    print('Comparison metric table of ' + original_hrtf_path + ' and ' + upsampled_hrtf_path)
    print(df)
    
    return metrics,threshold_bool,df


'''
hrtf_metrics.py. Functions that calculate metrics to numerically analyse differences between hrtfs
'''
import sys
import numpy as np
import scipy.signal as sn
from spatialaudiometrics import signal_processing as sp

def hrir2hrtf(hrir:np.array,fs,db_flag = 1):
    '''
    Converts all hrirs in a 3D array (location x ear x sample) into hrtfs. 
    
    :param hrir: HRIRs in a 3D array (location x ear x sample)
    :param fs: sample rate
    :param db_flag: if you want the spectra in dB rather than magnitude
    :return hrtfs: head related transfer functions (location x ear x sample)
    :return freqs: frequencies of the transfer function (length of hrtf)
    :return hrtfs_phase: phase information (location x ear x sample)
    '''
    sz          = [np.shape(hrir)[0],np.shape(hrir)[1],int(np.shape(hrir)[2]/2)]
    hrtfs       = np.empty(sz)
    hrtfs_phase = np.empty(sz)
    for l,loc in enumerate(hrir):
        for e,ear in enumerate(loc):
            spec, freqs, phase  = sp.calculate_spectrum(ear,fs,db_flag)
            hrtfs[l,e,:]        = spec
            hrtfs_phase[l,e,:]  = phase
            
    return hrtfs, freqs, hrtfs_phase

def calculate_lsd(tf1:np.array, tf2:np.array):
    '''
    Calculates the log spectral distortion between two transfer functions tf1 and tf2
    returns a list of values which is the lsd for each frequency

    :param tf1: First transfer function 
    :param tf2: Second transfer function to compare against the first
    :return lsd: Array of showing the lsd at each frequency point represented in th tfs
    '''
    lsd = 20*np.log10(tf1/tf2)
    return lsd

def calculate_lsd_across_freqs(tf1:np.array,tf2:np.array):
    '''
    Calculates the log spectral distortion across frequencies between two transfer functions tf1 and tf2
    
    :param tf1: First transfer function 
    :param tf2: Second transfer function to compare against the first
    :return lsd: Return a value that is the RMS across frequencies
    '''
    lsd = calculate_lsd(tf1,tf2)
    lsd = np.sqrt(np.mean(lsd**2))
    return lsd 

def calculate_lsd_across_locations(hrir1,hrir2,fs):
    '''
    Calculates the log spectral distortion across locations between two location matched hrirs only between 20 and 20000Hz
    
    :param hrir1: 3d array of the impulse response at each location x ear. Shape should be locations x ears x samples
    :param hrir2: 3d array of another impulse response at each location x ear. Shape should be locations x ears x samples
    :param fs: sample rate
    :param lsd: the mean lsd of across ears and locations
    :param lsd_mat: the lsd at each ear x location.
    '''
    if np.shape(hrir1)[2] != np.shape(hrir2)[2]:
        sys.exit('Error: The lengths of the impulse responses do not match, consider windowing them to be the same length')
    
    hrtfs1, freqs, hrtfs_phase = hrir2hrtf(hrir1,fs,db_flag = 0)
    hrtfs2, freqs, hrtfs_phase = hrir2hrtf(hrir2,fs,db_flag = 0)
    
    idx     = np.where((freqs >= 20) & (freqs <= 20000))[0] # This should be the same as the hrtfs should be matched in terms of length and they should have the same fs
    hrtfs1  = hrtfs1[:,:,idx]
    hrtfs2  = hrtfs2[:,:,idx]
    
    lsd_mat = np.empty(np.shape(hrir1)[0:2])
    for l,loc in enumerate(hrtfs1):
        for e,ear in enumerate(loc):
            lsd_mat[l,e] = calculate_lsd_across_freqs(hrtfs1[l,e,:],hrtfs2[l,e,:])
    
    lsd = np.mean(lsd_mat)
    return lsd,lsd_mat

def itd_estimator_maxiacce(hrir,fs):
    '''
    Calculates the ITD based on the MAXIACCe mode (transcribed from the itd_estimator in the AMTtoolbox 20/03/24, based on Andreopoulou et al. 2017)
    
    :param hrir: 3d array of the impulse response at each location x ear. Shape should be locations x ears x samples
    :return itd_s: ITD in seconds for each location
    :return itd_samps: ITD in samples for each location
    :return maxiacc: The max interaural cross correlation calculated 
    '''
    itd_samps   = list()
    maxiacc     = list()
    for loc in hrir:
        # Take the maximum absolute value of the cross correlation between the two ears to get the maxiacc
        correlation     = sn.correlate(np.abs(sn.hilbert(loc[0,:])),np.abs(sn.hilbert(loc[1,:])))
        maxiacc.append(np.max(np.abs(correlation)))
        idx_lag         = np.argmax(np.abs(correlation))
        itd_samps.append(idx_lag - np.shape(hrir)[2])
    itd_s = itd_samps/fs
    
    return itd_s,itd_samps,maxiacc

def ild_estimator_rms(hrir):
    '''
    Calculate the ILD by taking the rms of the impulse response at each ear and taking the difference
    
    :paran hrir: 3d array of the impulse response at each location x ear. Shape should be locations x ears x samples
    :return ild: ILD in dB for each location
    '''
    rms = np.sqrt(np.mean(hrir**2,axis = 2))
    ild = sp.mag2db(rms[:,0]) - sp.mag2db(rms[:,1])
    return ild

def calculate_itd_difference(hrtf1,hrtf2):
    '''
    Calculates the absolute difference in itd values between two hrtfs
    
    :param hrtf1: first hrtf (custom hrtf object)
    :param hrtf2: second hrtf (custom hrtf object)
    :return itd_diff: the average itd difference across locations in us 
    '''
    itd_s1,itd_samps,maxiacc = itd_estimator_maxiacce(hrtf1.hrir,hrtf1.fs)
    itd_s2,itd_samps,maxiacc = itd_estimator_maxiacce(hrtf2.hrir,hrtf2.fs)
    itd_diff = np.mean(np.abs(itd_s1-itd_s2)) * 1000000
    return itd_diff

def calculate_ild_difference(hrtf1,hrtf2):
    '''
    Calculates the absolute difference in ild values between two hrtfs
    
    :param hrtf1: first hrtf (custom hrtf object)
    :param hrtf2: second hrtf (custom hrtf object)
    :return ild_diff: the average ild difference across locations in dB 
    '''
    ild1        = ild_estimator_rms(hrtf1.hrir)
    ild2        = ild_estimator_rms(hrtf2.hrir)
    ild_diff    = np.mean(np.abs(ild1-ild2))
    return ild_diff

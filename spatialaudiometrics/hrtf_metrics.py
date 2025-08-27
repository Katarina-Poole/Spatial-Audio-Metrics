'''
hrtf_metrics.py. Functions that calculate metrics to numerically analyse differences between hrtfs
'''
import sys
import numpy as np
import scipy.signal as sn
import pandas as pd
from spatialaudiometrics import signal_processing as sp
from spatialaudiometrics import load_data as ld

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
    lsd = np.sqrt(np.mean(np.power(lsd,2)))
    return lsd 

def calculate_lsd_across_locations(hrir1,hrir2,fs,flow = 20,fhigh = 20000):
    '''
    Calculates the log spectral distortion across locations between two location matched hrirs only between 20 and 20000Hz
    
    :param hrir1: 3d array of the impulse response at each location x ear. Shape should be locations x ears x samples
    :param hrir2: 3d array of another impulse response at each location x ear. Shape should be locations x ears x samples
    :param fs: sample rate
    :param flow: lower frequency bound of lsd calculation in Hz
    :param fhigh: upper frequency bound of lsd calculation in Hz
    :param lsd: the mean lsd of across ears and locations
    :param lsd_mat: the lsd at each ear x location.
    '''
    if np.shape(hrir1)[2] != np.shape(hrir2)[2]:
        sys.exit('Error: The lengths of the impulse responses do not match, consider windowing them to be the same length')
    
    hrtfs1, freqs, hrtfs_phase = hrir2hrtf(hrir1,fs,db_flag = 0)
    hrtfs2, freqs, hrtfs_phase = hrir2hrtf(hrir2,fs,db_flag = 0)
    
    idx     = np.where((freqs >= flow) & (freqs <= fhigh))[0] # This should be the same as the hrtfs should be matched in terms of length and they should have the same fs
    hrtfs1  = hrtfs1[:,:,idx]
    hrtfs2  = hrtfs2[:,:,idx]
    
    lsd_mat = np.empty(np.shape(hrir1)[0:2])
    for l,loc in enumerate(hrtfs1):
        for e,ear in enumerate(loc):
            lsd_mat[l,e] = calculate_lsd_across_freqs(hrtfs1[l,e,:],hrtfs2[l,e,:])
    
    lsd = np.mean(lsd_mat)
    return lsd,lsd_mat

def calculate_lsd_across_locations_per_frequency(hrir1,hrir2,fs,flow = 20,fhigh = 20000):
    '''
    Calculates the log spectral distortion across locations between two location matched hrirs only between 20 and 20000Hz and includes frequency information
    
    :param hrir1: 3d array of the impulse response at each location x ear. Shape should be locations x ears x samples
    :param hrir2: 3d array of another impulse response at each location x ear. Shape should be locations x ears x samples
    :param fs: sample rate
    :param flow: lower frequency bound of lsd calculation in Hz
    :param fhigh: upper frequency bound of lsd calculation in Hz
    :param lsd_mat: the lsd at each ear x location x frequency.
    '''
    if np.shape(hrir1)[2] != np.shape(hrir2)[2]:
        sys.exit('Error: The lengths of the impulse responses do not match, consider windowing them to be the same length')
    
    hrtfs1, freqs, hrtfs_phase = hrir2hrtf(hrir1,fs,db_flag = 0)
    hrtfs2, freqs, hrtfs_phase = hrir2hrtf(hrir2,fs,db_flag = 0)
    
    idx     = np.where((freqs >= flow) & (freqs <= fhigh))[0] # This should be the same as the hrtfs should be matched in terms of length and they should have the same fs
    hrtfs1  = hrtfs1[:,:,idx]
    hrtfs2  = hrtfs2[:,:,idx]
    freqs   = freqs[idx]
    
    lsd_mat = np.empty((np.shape(hrir1)[0],np.shape(hrir1)[1],len(freqs)))
    for l,loc in enumerate(hrtfs1):
        for e,ear in enumerate(loc):
            lsd_mat[l,e,:] = calculate_lsd(hrtfs1[l,e,:],hrtfs2[l,e,:])
    
    return lsd_mat,freqs

def itd_estimator_maxiacce(hrir,fs, upper_cut_freq = 3000, filter_order = 10):
    '''
    Calculates the ITD based on the MAXIACCe mode (transcribed from the itd_estimator in the AMTtoolbox 20/03/24, based on Andreopoulou et al. 2017)
    Low passes the hrir
    :param hrir: 3d array of the impulse response at each location x ear. Shape should be locations x ears x samples
    :param fs: sample rate
    :param upper_cut_freq: the upper cut off point for the low pass filter 
    :param filter order: the filter order for the low pass filter
    :return itd_s: ITD in seconds for each location
    :return itd_samps: ITD in samples for each location
    :return maxiacc: The max interaural cross correlation calculated 
    '''
    itd_samps   = list()
    maxiacc     = list()
    wn          = upper_cut_freq/(fs/2)
    b,a         = sn.butter(filter_order,wn)

    for loc in hrir:
        # Filter the hrir
        loc_l = sn.lfilter(b,a,loc[0,:])
        loc_r = sn.lfilter(b,a,loc[1,:])
        # Take the maximum absolute value of the cross correlation between the two ears to get the maxiacc
        correlation     = sn.correlate(np.abs(sn.hilbert(loc_l)),np.abs(sn.hilbert(loc_r)))
        maxiacc.append(np.max(np.abs(correlation)))
        idx_lag         = np.argmax(np.abs(correlation))
        itd_samps.append(idx_lag - np.shape(hrir)[2])
    itd_s = itd_samps/fs
    
    return itd_s,itd_samps,maxiacc

def itd_estimator_threshold(hrir,fs,thresh_level = -10, upper_cut_freq = 3000, filter_order = 10):
    '''
    Calculates the ITD based on the threshold mode (transcribed from the itd_estimator in the AMTtoolbox 20/03/24, based on Andreopoulou et al. 2017)
    with parameters used by the SONICOM dataset to remove the ITD. 
    :param hrir: 3d array of the impulse response at each location x ear. Shape should be locations x ears x samples
    :param fs: sample rate
    :param upper_cut_freq: the upper cut off point for the low pass filter 
    :param filter order: the filter order for the low pass filter
    :return itd_s: ITD in seconds for each location
    :return itd_samps: ITD in samples for each location
    '''
    itd_samps   = list()
    itd_index   = np.zeros([np.shape(hrir)[0],np.shape(hrir)[1]])
    wn          = upper_cut_freq/(fs/2)
    b,a         = sn.butter(filter_order,wn)

    for p,loc in enumerate(hrir):
        itd_lr_samps = list()
        for e in range(2):
            # Filter the hrir
            filt_loc    = sn.lfilter(b,a,loc[e,:])
            in_db       = 0.5*sp.mag2db(np.square(filt_loc))
            thresh_value = max(in_db) + thresh_level
            idx         = np.where(in_db > thresh_value)[0][0]
            itd_lr_samps.append(idx)
            itd_index[p,e] = int(idx)
        itd_samps.append(itd_lr_samps[0] - itd_lr_samps[1])
    
    itd_s = itd_samps/fs

    return itd_s, itd_samps, itd_index


def ild_estimator_rms(hrir):
    '''
    Calculate the ILD by taking the rms of the impulse response at each ear and taking the difference
    
    :paran hrir: 3d array of the impulse response at each location x ear. Shape should be locations x ears x samples
    :return ild: ILD in dB for each location
    '''
    rms = np.sqrt(np.mean(np.power(hrir,2),axis = 2))
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

def calculate_ild_difference(hrtf1,hrtf2, average = True, type = 'hrtf'):
    '''
    Calculates the absolute difference in ild values between two hrtfs
    
    :param hrtf1: first hrtf (custom hrtf object)
    :param hrtf2: second hrtf (custom hrtf object)
    :return ild_diff: the average ild difference across locations in dB 
    '''
    if type == 'hrtf':
        ild1        = ild_estimator_rms(hrtf1.hrir)
        ild2        = ild_estimator_rms(hrtf2.hrir)
    elif type == 'dtf':
        ild1        = ild_estimator_rms(hrtf1.dtf_ir)
        ild2        = ild_estimator_rms(hrtf2.dtf_ir)

    if average:
        ild_diff    = np.mean(np.abs(ild1-ild2))
    else:
        ild_diff    = ild1-ild2
    return ild_diff


def hrtf2dtf(hrtf, f1=20, f2=20000, atten = 20, rms = False):
    '''
    Extracts the directional transfer function (DTF) and common transfer function (CTF) from HRTF data.

    :param hrtf: hrtf to be converted to dtf and ctf (custom hrtf object)
    :param f1: Lower frequency bound of filtering in Hz, default = 50Hz in AMT but 20Hz in SAM
    :param f2: Upper frequency bound of filtering in Hz, default = 18 kHz in AMT but 20kHz in SAM
    :param atten: Attenuation in dB to apply to the CTF to avoid clipping, default = 20dB
    :param weights: not supported at the moment.
    :param rms: If True, evaluate CTF magnitude spectrum by RMS of linear magnitude spectra, equivalent to diffuse-field compensation instead of default log-manitude spectra
    :return dtf: Directional transfer function (DTF)
    :return ctf: Common transfer function (CTF)
    
    :Original function from AMTtoolbox (Author: Robert Baumgartner (2010)), transcribed to Python. 
    '''
    fs          = hrtf.fs
    hrtfmtx     = np.transpose(hrtf.hrir, (2,0,1))
    n           = hrtfmtx.shape[0]
    nfft        = n
    
    # Frequency bounds
    df          = fs/nfft # Frequency resolution
    f           = np.arange(0,fs,df)
    idx         = ((f >= f1) & (f <= f2))
    idx[int(nfft/2)+1:] = idx[1:int(nfft/2)][::-1]
    
    # CTF calculation
    hrtff = np.fft.fft(hrtfmtx, n = nfft, axis=0)  # FFT along the first axis (frequency)
    
    if rms:
        ctffavg = np.sqrt(np.mean(np.square(np.abs(hrtff)), axis=1))
    else:
        ctffavg = np.mean(np.log(np.abs(hrtff)+np.finfo(float).eps),axis = 1)
    
    # Force minimum phase
    ctfflog = np.mean(np.log(np.abs(hrtff)+np.finfo(float).eps), axis=1)
    ctfcep = np.fft.ifft(ctfflog,n = nfft, axis = 0) # flip acausal part to causal part or simply multiply
    ctfcep[int(nfft/2)+1:nfft+1,:] = 0
    ctfcep[1:int(nfft/2),:] = 2*ctfcep[1:int(nfft/2),:]    # causal part by 2 (due to symmetry)
    ctfflog = np.fft.fft(ctfcep,nfft,axis = 0)
    ctfp = np.exp(ctfflog)

    # Get complex spectrum
    if rms:
        ctff = ctffavg * np.exp(1j * np.angle(ctfp))
    else:
        ctff = np.exp(ctffavg) * np.exp(1j * np.angle(ctfp))
    
    # Get IR
    ctfmtx = np.real(np.fft.ifft(ctff, n = nfft, axis = 0))

    # DTF Calculation
    dtff = np.copy(hrtff)
    dtff[idx, :, :] = hrtff[idx, :, :]/(ctff[idx,  np.newaxis,:])
    dtfmtx = np.real(np.fft.ifft(dtff, n=nfft, axis=0))

    # Attenuate to avoid clipping
    ctfmtx = ctfmtx/np.power(10, atten/20)
    dtfmtx = dtfmtx/np.power(10, atten/20)

    dtf = np.transpose(dtfmtx, (1,2,0))
    ctf = np.transpose(ctfmtx, (1,0))
    ctf = ctf[np.newaxis,:,:]
    
    return dtf,ctf

def generate_table_difference_hrtfs(hrtf1,hrtf2,type = 'hrtf'):
    '''
    Generates a table that numerical calculates differences between two HRTFs (that are equal in window size and sample rate and locations)
    This will do the location matching for you
    :param hrtf1: SAM HRTF object (usually the synthetic hrtf)
    :params hrtf2: SAM HRtF object (usually the measured hrtf)
    '''
    hrtf1, hrtf2        = ld.match_hrtf_locations(hrtf1,hrtf2)

    df                  = pd.DataFrame()
    df['itd_diff_us']   = (hrtf1.itd_s-hrtf2.itd_s)*1000000
    if type == 'hrtf':
        df['ild_diff_db']   = calculate_ild_difference(hrtf1,hrtf2,False)
        lsd,lsd_mat         = calculate_lsd_across_locations(hrtf1.hrir,hrtf2.hrir,hrtf1.fs,50,18000)
    elif type == 'dtf':
        df['ild_diff_db']   = calculate_ild_difference(hrtf1,hrtf2,False, type = 'dtf')
        lsd,lsd_mat         = calculate_lsd_across_locations(hrtf1.dtf_ir,hrtf2.dtf_ir,hrtf1.fs,50,18000)

    df['az']            = hrtf1.locs[:,0]
    df['el']            = hrtf1.locs[:,1]
    df['lsd_l']         = lsd_mat[:,0]
    df['lsd_r']         = lsd_mat[:,1]

    return df

def generate_table_difference_lsd_freq_hrtfs(hrtf1,hrtf2, type = 'hrtf'):
    '''
    Generates a table that numericall cualtes the LSD for each frequency
    '''
    if type == 'hrtf':
        lsd_mat,freqs = calculate_lsd_across_locations_per_frequency(hrtf1.hrir,hrtf2.hrir,hrtf1.fs)
    elif type == 'dtf':
        lsd_mat,freqs = calculate_lsd_across_locations_per_frequency(hrtf1.dtf_ir,hrtf2.dtf_ir,hrtf1.fs)
    
    dfs = list()
    ear_name = ['left','right']
    for loc in range(np.shape(lsd_mat)[0]):
        ear_df = list()
        for ear in range(np.shape(lsd_mat)[1]):
            df = pd.DataFrame()
            df['lsd'] = lsd_mat[loc,ear,:]
            df['freqs'] = freqs
            df['ear'] =  ear_name[ear]
            df['az'] = hrtf1.locs[loc,0]
            df['el'] = hrtf1.locs[loc,1]
            ear_df.append(df)
        dfs.append(pd.concat(ear_df,axis = 0))
    out_df = pd.concat(dfs,axis = 0)
    loc_out_df = out_df.groupby(['freqs','ear','az','el']).lsd.apply(sp.rms).reset_index()
    out_df = out_df.groupby(['freqs','ear']).lsd.apply(sp.rms).reset_index()

    return out_df,loc_out_df
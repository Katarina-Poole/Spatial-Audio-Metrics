'''
hrtf_metrics.py. Functions that calculate metrics to numerically analyse differences between hrtfs
'''
import numpy as np
from scipy.fft import fft, fftfreq

def mag2db(x):
    '''
    Convert values from magnitude to dB using 20log10
    :param x: float value
    :return y: float value (dB)
    '''
    y = 20*np.log10(x)
    return y

def calculate_spectrum(x:np.array,fs,db_flag = 1):
    '''
    Converts a time domain signal (such as an impulse reponse) to the frequency domain (such as a transfer function). The default is to return the output in dB
    :param x: 1D numpy array
    :param fs: sample rate of the signal (e.g. 48000)
    :param db_flag: if you want the spectra in dB rather than magnitude
    :return spec: spectrum (e.g. transfer function)
    :return freqs: the frequencies for each value in the spectrum
    :return phase: phase
    '''
    n       = len(x) # samples
    t       = 1.0/fs # time in seconds between samples
    freqs   = fftfreq(n,t)[:n//2]
    y       = fft(x)
    amp     = np.abs(y[0:n//2])
    phase   = np.imag(y[0:n//2])
    db      = mag2db(amp)
    if db_flag == 1:
        spec = db
    else:
        spec = amp
    return spec, freqs, phase
    

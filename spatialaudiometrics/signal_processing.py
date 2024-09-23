'''
signal_processing.py. Generic signal processing functions
'''
import numpy as np
from scipy.fft import fft, fftfreq
import scipy

def mag2db(x):
    '''
    Convert values from magnitude to dB using 20log10
    
    :param x: float value
    :returns y: float value (dB)
    '''
    y = 20*np.log10(x)
    return y

def db2mag(x):
    '''
    Convert values from dB to magnitude
    
    :param x: float value (dB)
    :returns y: float value 
    '''
    y = np.float_power(10,x/20)
    return y

def calculate_spectrum(x:np.array,fs,db_flag = 1):
    '''
    Converts a time domain signal (such as an impulse reponse) to the frequency domain (such as a transfer function). The default is to return the output in dB
    
    :param x: 1D numpy array
    :param fs: sample rate of the signal (e.g. 48000)
    :param db_flag: if you want the spectra in dB rather than magnitude
    :returns spec: spectrum (e.g. transfer function)
    :returns freqs: the frequencies for each value in the spectrum
    :returns phase: phase
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

def create_wavelet(frequency,fs,oscillations_per_side = 7):
    '''
    Creates a morlet wavelet (for wavelet decomposition)
    :param frequency: frequency of the wavelet
    :param fs: sample rate
    :param oscillations_per_side: how wide the wavelet is in number of oscillations
    :return: return the real and imaginary part of the wavelet
    '''
    period          = 1/frequency
    double_period   = 2/frequency

    ts              = np.arange(-(period*oscillations_per_side),period*oscillations_per_side,1/fs)
    wavelet_cos     = np.cos(2*np.pi*frequency*ts)
    wavelet_sin     = np.sin(2*np.pi*frequency*ts)
    wavelet_gauss   = (np.power(double_period,-0.5)*np.power(np.pi,-0.25))*np.exp(np.power(-ts,2)/(2*np.power(double_period,2)))

    wavelet_real    = wavelet_cos * wavelet_gauss
    wavelet_imag    = wavelet_sin * wavelet_gauss

    return wavelet_real, wavelet_imag

def wavelet_decomposition(sig,fs,freq_steps = 1, freq_min = 0.5, freq_max = 20000):
    '''
    Runs wavelet decomposition on the signal
    Try and use FWHM to deinfe the number of cycles

    https://www.sciencedirect.com/science/article/pii/S1053811919304409
    :param sig: signal you want to decompose
    :param fs: sample rate
    :param freq_steps: the step size of frequencies to be decomposed (i.e. 1 is every 1hz step)
    :param freq_min: the minimum frequency you want
    :param freq_max: the maximum frequency you want
    :returns: mag. phase and frequencies of the decomposition
    '''
    freqs           = np.arange(freq_min,freq_max,freq_steps)
    num_wavelets    = len(freqs)

    mag             = np.zeros([num_wavelets,len(sig)])
    phase           = np.zeros([num_wavelets,len(sig)])

    for i,f in enumerate(freqs):
        wavelet_real,wavelet_imag = create_wavelet(f,fs)
        real        = scipy.signal.convolve(sig,wavelet_real,mode = 'same')
        imag        = scipy.signal.convolve(sig,wavelet_imag,mode = 'same')
        mag[i,:]    = np.sqrt((real*real)+(imag*imag))
        phase[i,:]  = np.arctan2(imag,real)

    return mag, phase, freqs

def rms(x):
    rms = np.sqrt(np.mean(np.power(x,2)))
    return rms

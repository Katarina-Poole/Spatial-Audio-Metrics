'''
signal_processing.py. Functions that perform signal processing
'''

import numpy as np

def calculate_lsd(tf1:np.array, tf2:np.array):
    '''
    Calculates the log spectral distortion between two transfer functions tf1 and tf2
    returns a list of values which is the lsd for each frequency

    :param tf1: First transfer function 
    :param tf2: Second transfer function to compare against the first
    '''
    lsd = 20*np.log10(tf1/tf2)
    return lsd

def calculate_lsd_across_freqs(tf1:np.array,tf2:np.array):
    '''
    Calculates the log spectral distortion across frequencies between two transfer functions tf1 and tf2
    returns a list of values which is the lsd for each frequency

    :param tf1: First transfer function 
    :param tf2: Second transfer function to compare against the first
    '''
    lsd = calculate_lsd(tf1,tf2)
    lsd = np.sqrt(np.mean(lsd**2))
    return lsd 
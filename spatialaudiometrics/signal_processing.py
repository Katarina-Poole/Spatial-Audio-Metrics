'''
signal_processing.py. Functions that loads in example data

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

def calculate_lsd(tf1,tf2):
    '''
    Calculates the log spectral distortion between two transfer functions tf1 and tf2
    returns a list of values which is the lsd for each frequency
    '''
    lsd = 20*np.log10(tf1/tf2)
    return lsd

def calculate_lsd_across_freqs(tf1,tf2):
    '''
    Calculates the log spectral distortion across frequencies between two transfer functions tf1 and tf2
    returns a list of values which is the lsd for each frequency
    '''
    lsd = calculate_lsd(tf1,tf2)
    lsd = np.sqrt(np.mean(lsd**2))
    return lsd 
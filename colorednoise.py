'''
Generate Dicrete Colored Noise

python / numpy implementation of 
Kasdin, N.J., Walter, T., "Discrete simulation of power law noise 
[for oscillator stability evaluation]," 
Frequency Control Symposium, 1992. 46th., Proceedings of the 1992 IEEE, 
pp.274,283, 27-29 May 1992
'''

# author: Julia Leute

import numpy as np

def noiseGen(nr, Qd, b):

    ''' Generates discrete colored noise
    required inputs: 
        nr (number of points, must be power of two)
        Qd (discrete variance)
        b (slope of the noise) '''

    mhb = -b/2.0
    Qd = np.sqrt(Qd)
    # fill sequence wfb with white noise
    wfb = np.zeros(nr*2)
    wfb[:nr] = np.random.normal(0, Qd, nr)
    # generate the coefficients hfb
    hfb = np.zeros(nr*2)    
    hfb[0] = 1.0
    indices = np.arange(nr-1)
    hfb[1:nr] = (mhb+indices)/(indices+1.0)
    hfb[:nr] = np.multiply.accumulate(hfb[:nr])
    # discrete Fourier transforms of white noise and coefficients,
    # multiplication of resulting complex vectors, 
    # inverse Fourier transform
    colorednoise = np.fft.irfft(np.fft.rfft(wfb)*np.fft.rfft(hfb))[:nr]
    return colorednoise

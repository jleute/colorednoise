'''
Generate Dicrete Colored Noise

python / numpy implementation of 
Kasdin, N.J., Walter, T., "Discrete simulation of power law noise 
[for oscillator stability evaluation]," 
Frequency Control Symposium, 1992. 46th., Proceedings of the 1992 IEEE, 
pp.274,283, 27-29 May 1992
http://dx.doi.org/10.1109/FREQ.1992.270003
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

def phase_psd_from_qd(qd, b, tau0):
    """ return phase power spectral density coefficient g_b from QD
    
        Colored noise generated with (qd, b, tau0) parameters will
        show a phase power spectral density of
        S_x(f) = Phase_PSD(f) = g_b * f^b
        
        Kasdin & Walter eqn (39)
    """
    g_b = qd*2.0*pow(2.0*np.pi, b)*pow(tau0, b+1.0)
    return g_b

def frequency_psd_from_qd(qd, b, tau0):
    """ return frequency power spectral density coefficient h_a from QD
        
        Colored noise generated with (qd, b, tau0) parameters will
        show a frequency power spectral density of
        
        S_y(f) = Frequency_PSD(f) = h_a * f^a
        where the slope a comes from the phase PSD slope b:
        a = b + 2
        
        Kasdin & Walter eqn (39)
    """
    a = b + 2.0
    h_a = qd*2.0*pow(2.0*np.pi, a)*pow(tau0, a-1.0)
    return h_a
    
def adev_from_qd(qd, b, tau0, tau):
    """ prefactor for Allan deviation from QD and slope
    
        Colored noise generated with (qd, b, tau0) parameters will
        show an Allan variance of:
        
        AVAR = prefactor * h_a * tau^c
        
        where a = b+2 is the slope of the frequency PSD.
        and h_a is the frequency PSD prefactor S_y(f) = h_a * f^a
        
        The relation between a, b, c is:
        a   b   c(AVAR) c(MVAR)
        -----------------------
        -2  -4   1       1
        -1  -3   0       0
         0  -2  -1      -1
        +1  -1  -2      -2
        +2  -2  -2      -3
    
        Coefficients from:
        S. T. Dawkins, J. J. McFerran and A. N. Luiten, 
        "Considerations on the measurement of the stability of 
        oscillators with frequency counters," in 
        IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 
        vol. 54, no. 5, pp. 918-925, May 2007. doi: 10.1109/TUFFC.2007.337
    """
    g_b = phase_psd_from_qd(qd,b,tau0)
    f_h = 0.5/tau0
    if b == 0:
        coeff = 3.0*f_h / (4.0*pow(np.pi,2)) # E, White PM, tau^-1
    elif b == -1:
        coeff = (1.038+3*np.log(2.0*np.pi*f_h*tau))/(4.0*pow(np.pi,2))# D, Flicker PM, tau^-1
    elif b == -2:
        coeff = 0.5 # C, white FM,  1/sqrt(tau)
    elif b == -3:
        coeff = 2*np.log(2) # B, flicker FM,  constant ADEV
    elif b == -4:
        coeff = 2.0*pow(np.pi,2)/3.0 #  A, RW FM, sqrt(tau)

    return np.sqrt(coeff*g_b*pow(2.0*np.pi,2))

def mdev_from_qd(qd, b, tau0, tau):
    """ prefactor for Modified Allan deviation from QD and slope
    
        Colored noise generated with (qd, b, tau0) parameters will
        show an Modified Allan variance of:
        
        MVAR = prefactor * h_a * tau^c
        
        where a = b+2 is the slope of the frequency PSD.
        and h_a is the frequency PSD prefactor S_y(f) = h_a * f^a
        
        The relation between a, b, c is:
        a   b   c(AVAR) c(MVAR)
        -----------------------
        -2  -4   1       1
        -1  -3   0       0
         0  -2  -1      -1
        +1  -1  -2      -2
        +2  -2  -2      -3

        Coefficients from:
        S. T. Dawkins, J. J. McFerran and A. N. Luiten, 
        "Considerations on the measurement of the stability of 
        oscillators with frequency counters," in 
        IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 
        vol. 54, no. 5, pp. 918-925, May 2007. doi: 10.1109/TUFFC.2007.337
    """
    g_b = phase_psd_from_qd(qd,b,tau0)
    f_h = 0.5/tau0
    if b == 0:
        coeff = 3.0/(8.0*pow(np.pi,2)) # E, White PM, tau^-{3/2}
    elif b == -1:
        coeff = (24.0*np.log(2)-9.0*np.log(3))/8.0/pow(np.pi,2) # D, Flicker PM, tau^-1
    elif b == -2:
        coeff = 0.25 # C, white FM,  1/sqrt(tau)
    elif b == -3:
        coeff = 2.0*np.log(3.0*pow(3.0,11.0/16.0)/4.0) # B, flicker FM,  constant MDEV
    elif b == -4:
        coeff = 11.0/20.0*pow(np.pi,2) #  A, RW FM, sqrt(tau)

    return np.sqrt(coeff*g_b*pow(2.0*np.pi,2))

'''
KVAL.SIGNAL.DESPIKE

Routines for computing FFTs and power spectral densities.

VERY preliminary - hastily copied from the old oyv.num.spec module
and not really reqritten for xarray
'''

import numpy as np
import xarray as xr
import scipy.signal as ss
import numpy.fft as fft



def apply_window(y, window='none', axis = 0):
    '''
    Applies the chosen window convolution.
    '''

    L = np.shape(y)[axis]
    x = np.arange(-L/2,L/2)

    if window=='tri':
        w = 1-(abs(x))/float(L)
    elif window=='han':
        w = np.sin(np.pi*(x+L/2)/L)**2
    elif window == 'bman':
        w = 0.42 - 0.5*np.cos(2*np.pi*(x+L/2)/L) +\
            0.08 * np.cos(2*2*np.pi*(x+L/2)/L)
    else:
        w = np.ones(L)

   # dxobj = [None for i in range(y.ndim)]
   # dxobj[axis] = slice(None)

  #  print(dxobj)
    Y = y * w[:, np.newaxis]#[dxobj]

    return Y
    #Check: OK




def detrend_series(y, axis = 0):
    '''
    scipy.signal.detrend that works for complex values.
    '''

    if y.ndim < (axis+1):
        estr = 'Axis ('+ str(axis) +\
        ') exceeds array dimensions ('+str(y.ndim)+')'

        raise ValueError(estr)

    y_detr = ss.detrend(y.real, axis = axis)

    if np.iscomplex(y).any():
        y_imag_detr = ss.detrend(y.imag, axis = axis)
        y_detr += 1j*y_imag_detr

    return y_detr
    # Checked for 1d arrays..





def wfft(y, t,  n , ov = 0.66, axis = 0, window = 'bman',
            ndt_ret = False, tint_ret = False, detrend = True):
    '''
    Windowed fft.
    y: Data field
    t: Space/time coordinate
    n: Number of windows
    ov: Overlap
    axis: Axis over which to perform fft
    window: Window shape (see below)
    ndt_ret: Return N*dt, window length in units of t.
    tint_ret: Returns the start and end location of each t segment.

    Returns FFT, f, T where f is the frequency/wavenumber (cycles per unit),
    and T is the average t at each transform interval. FFT is the unflattened
    fft array with layers in dimension 0. If a windowing function is
    used, a scaine factor is applied (see Emery/Thompson).

    Option ndt_ret returns the window length in units of t. (Should be
    applied to power spectra).

    Allowed windows are "bman" (Blackman), "han" (Hanning) and
    "none".

    Only accepts arrays that don't contain NaNs or masked points.
    '''

    # Scaling factor:
    if window == 'bman':
        w = 3.283
    elif window == 'han':
        w = 8/3.0
    elif window == 'none':
        w = 1
    else:
        raise ValueError("Unknown window value. Options: 'bman',"+
        " 'han' and 'none'.")

    dims = y.ndim
    L = np.shape(y)[axis]

    if np.ma.is_masked(y):
        if y.mask.any():
            raise ValueError("Error: y contains masked points.")
    if np.isnan(y).any():
        raise ValueError("Error: y contains NaNs.")

    if len(t) != L:
        raise ValueError("Error: t must have the same length as the "\
                         "fft'ed axis of y")

    dt = np.ma.mean(np.diff(t))

    if (np.diff(t) == dt).all() == False: #
        print('NB: t grid is uneven. Might want to interpolate.')
    wl_b = L/(n-ov*(n-1)) # Window length before adjusting
    wl = int(round(wl_b/2)*2) # Window length after adjusting

    if n == 1:
        ov_n = ov
        wl = L
    else:
        ov_n = (wl*n-L)/float(L-wl) # True overlap after
        # adjusting(actually, the inverse..)
    dn = wl/(1+ov_n) # Length between windows
    ft = fft.fftshift(fft.fftfreq(wl, d=dt))
    df = ft[1]-ft[0]

    # FFT is a (dims+1)d array with n in dimension 0
    FFT = []
    T = []
    tint  = []
    for nn in np.arange(0,n):
        sl = [slice(None,)]*dims
        t_sl = slice(int(nn*dn), int(nn*dn)+wl)
        tint = tint + [[t[t_sl.start], t[t_sl.stop-1]]]
        sl[axis] = t_sl

        y_in = y[sl[0]] ### !!!
        if detrend:
            y_in = detrend_series(y_in, axis = axis)

        fft_nn = (np.sqrt(w)
                  *fft.fftshift(fft.fft(
                      apply_window(y_in,
                      window = window, axis = axis), \
                      axis = axis), axes = (axis,)))

        FFT = FFT + [fft_nn]
        T = T + [np.mean(t[t_sl])]

    if ndt_ret:
        ndt = wl * dt
        if tint_ret:
            return np.array(FFT), np.array(ft), np.array(T), ndt, tint
        else:
            return np.array(FFT), np.array(ft), np.array(T), ndt
    else:
        if tint_ret:
            return np.array(FFT), np.array(ft), np.array(T), tint
        else:
            return np.array(FFT), np.array(ft), np.array(T)



def psd(y, t, n, ov = 0.66, axis = 0, window = 'bman', detrend = True):
    ''''
    PSD estimate of 1d or 2d array. Optional block averaging.

    Returns psd array (S) and frequency vector (ft).

    For real data, the one-sided spectrum is returned (including a
    factor of 2). For complex data, the two-sided spectrum is
    returned.
    '''

    L = len(t)
    dt = t[1]-t[0]
    FFT, ft, T, ndt = wfft(y, t, n, axis = axis, ndt_ret = True,
                           ov = ov, window = window, detrend = detrend)
    FFT = FFT * dt
    sl_list = [slice(None)] * y.ndim

    if np.iscomplex(y).any():
        s_ = (FFT*np.conj(FFT))/ndt
    else:


        rhslice = slice(int(len(ft)//2+1), None)
        sl_list[axis] = rhslice
        s_ = (2*(FFT*np.conj(FFT))/ndt)
        ft = ft[rhslice]

    s = np.ma.mean(s_.real, axis = 0)[rhslice]
    return s, ft

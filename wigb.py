#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 17:03:48 2019

@author: nephilim
"""

import numpy as np
import matplotlib.pyplot as plt



def insert_zeros(trace, tt=None):
    """Insert zero locations in data trace and tt vector based on linear fit"""

    if tt is None:
        tt = np.arange(len(trace))

    # Find zeros
    zc_idx = np.where(np.diff(np.signbit(trace)))[0]
    x1 = tt[zc_idx]
    x2 = tt[zc_idx + 1]
    y1 = trace[zc_idx]
    y2 = trace[zc_idx + 1]
    a = (y2 - y1) / (x2 - x1)
    tt_zero = x1 - y1 / a

    # split tt and trace
    tt_split = np.split(tt, zc_idx + 1)
    trace_split = np.split(trace, zc_idx + 1)
    tt_zi = tt_split[0]
    trace_zi = trace_split[0]

    # insert zeros in tt and trace
    for i in range(len(tt_zero)):
        tt_zi = np.hstack(
            (tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack(
            (trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi


def wiggle_input_check(data, tt, xx, sf, verbose):
    ''' Helper function for wiggle() and traces() to check input
    '''

    # Input check for verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    # Input check for data
    if type(data).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")

    if len(data.shape) != 2:
        raise ValueError("data must be a 2D array")

    # Input check for tt
    if tt is None:
        tt = np.arange(data.shape[0])
        if verbose:
            print("tt is automatically generated.")
            print(tt)
    else:
        if type(tt).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(tt.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")

    # Input check for xx
    if xx is None:
        xx = np.arange(data.shape[1])
        if verbose:
            print("xx is automatically generated.")
            print(xx)
    else:
        if type(xx).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(xx.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")
        if verbose:
            print(xx)

    # Input check for streth factor (sf)
    if not isinstance(sf, (int, float)):
        raise TypeError("Strech factor(sf) must be a number")

    # Compute trace horizontal spacing
    ts = np.min(np.diff(xx))

    # Rescale data by trace_spacing and strech_factor
    data_max_std = np.max(np.std(data, axis=0))
    data = data / data_max_std * ts * sf

    return data, tt, xx, ts


def wiggle(data, tt=None, xx=None, color='k', sf=0.15, verbose=False):
    '''Wiggle plot of a sesimic data section
    Syntax examples:
        wiggle(data)
        wiggle(data, tt)
        wiggle(data, tt, xx)
        wiggle(data, tt, xx, color)
        fi = wiggle(data, tt, xx, color, sf, verbose)
    Use the column major order for array as in Fortran to optimal performance.
    The following color abbreviations are supported:
    ==========  ========
    character   color
    ==========  ========
    'b'         blue
    'g'         green
    'r'         red
    'c'         cyan
    'm'         magenta
    'y'         yellow
    'k'         black
    'w'         white
    ==========  ========
    '''

    # Input check
    data, tt, xx, ts = wiggle_input_check(data, tt, xx, sf, verbose)

    # Plot data using matplotlib.pyplot
    Ntr = data.shape[1]

    ax = plt.gca()
    for ntr in range(Ntr):
        trace = data[:, ntr]
        offset = xx[ntr]

        if verbose:
            print(offset)

        trace_zi, tt_zi = insert_zeros(trace, tt)
        ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                         where=trace_zi >= 0,
                         facecolor=color)
        ax.plot(trace_zi + offset, tt_zi, color)

    ax.set_xlim(xx[0] - ts, xx[-1] + ts)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()


if __name__=='__main__':
    from matplotlib import pyplot
    from scipy import interpolate 
    import skimage.transform
    import WiggleDataNormalized
    Data_=WiggleDataNormalized.WiggleDataNormalized(ProfileNoise)
    wiggle(Data_,xx=np.arange(111),color='k',sf=0.3,verbose=False)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,111,12))
    ax.set_xticklabels([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5])
    ax.set_yticks(np.linspace(0,256,7))
    ax.set_yticklabels([0,10,20,30,40,50,60])
    pyplot.xlabel('Scan (m)')
    pyplot.ylabel('Time (ns)')
    pyplot.savefig('NoiseWiggle.png',dpi=1000)
    
    pyplot.figure()
    NewData_Residual=skimage.transform.resize(NewData_Residual,(256,111),mode='symmetric')
    Data_=WiggleDataNormalized.WiggleDataNormalized(NewData_Residual)
    wiggle(Data_,xx=np.arange(111),color='k',sf=0.3,verbose=False)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,111,12))
    ax.set_xticklabels([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5])
    ax.set_yticks(np.linspace(0,256,7))
    ax.set_yticklabels([0,10,20,30,40,50,60])
    pyplot.xlabel('Scan (m)')
    pyplot.ylabel('Time (ns)')
    pyplot.savefig('DenoiseWiggle.png',dpi=1000)
    
    pyplot.figure()
    data=Data_-NewData_Residual
    data=WiggleDataNormalized.WiggleDataNormalized(data)
    wiggle(data,xx=np.arange(111),color='k',sf=0.4,verbose=False)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,111,12))
    ax.set_xticklabels([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5])
    ax.set_yticks(np.linspace(0,256,7))
    ax.set_yticklabels([0,10,20,30,40,50,60])
    pyplot.xlabel('Scan (m)')
    pyplot.ylabel('Time (ns)')
    pyplot.savefig('ResidualWiggle.png',dpi=1000)


# import numpy as np
# import matplotlib.pyplot as plt
# from fatiando.seismic import conv
# from fatiando.vis.mpl import seismic_wiggle

# pyplot.figure(figsize=(40,10))
# Data=data
# Data_=WiggleDataNormalized.WiggleDataNormalized(Data)
# seismic_wiggle(Data_,scale=0.8, color='k')
# pyplot.savefig('data.svg')
# pyplot.figure()
# pyplot.imshow(Data_,vmin=np.min(Data_),vmax=np.max(Data_),extent=extent,cmap='gray')

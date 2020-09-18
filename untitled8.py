#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 22:35:45 2020

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot,cm
import T_PowerGain
import skimage.transform
import scipy.io as scio

def CalculationSNR(Image,Noise):
    frac_up=np.sum(Image**2)
    frac_down=np.sum((Image-Noise)**2)
    SNR=10*np.log10(frac_up/frac_down)
    return SNR

Profile=scio.loadmat('./normal1.mat')['data1']
ProfileNoiseGain=T_PowerGain.tpowGain(Profile,np.arange(5001)/4,0.9)
ProfileGain=ProfileNoiseGain.copy()
ProfileNoiseGain=skimage.transform.resize(ProfileNoiseGain,(256,256),mode='edge')        
ProfileNoiseGain=ProfileNoiseGain+1e3*(np.random.random(ProfileNoiseGain.shape)-0.5)
ProfileNoiseGain[45:155,190:193]=ProfileNoiseGain[45:155,190:193]+1e2*(np.random.random(ProfileNoiseGain[45:155,190:193].shape)-0.5)

ProfileNoiseGain[:,90:93]=ProfileNoiseGain[:,90:93]+1e2*(np.random.random(ProfileNoiseGain[:,90:93].shape)-0.5)

ProfileNoiseGain[:,100:103]=ProfileNoiseGain[:,100:103]+2e2*(np.random.random(ProfileNoiseGain[:,100:103].shape)-0.5)

ProfileNoiseGain[:,150:153]=ProfileNoiseGain[:,150:153]-1.2e2*(np.random.random(ProfileNoiseGain[:,150:153].shape)-0.5)
ProfileNoiseGain=skimage.transform.resize(ProfileNoiseGain,(5001,91),mode='edge')        
# np.save('Noise_Test.npy',ProfileNoiseGain)
CalculationSNR(ProfileGain,ProfileNoiseGain)
scio.savemat('Denoise.mat',{'data':ProfileNoiseGain})
pyplot.imshow(ProfileNoiseGain,vmin=np.min(ProfileNoiseGain),vmax=np.max(ProfileNoiseGain),extent=(0,1,1,0),cmap=cm.seismic)
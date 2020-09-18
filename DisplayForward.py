#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:29:07 2020

@author: nephilim
"""

import numpy as np
import skimage.transform
import T_PowerGain
import DataNormalized
from matplotlib import pyplot,cm


# Read Target Profile
ProfileTarget=np.load('./GPR_Modelling/ProfileAutoEncoder/33_iter_record_0_comp.npy')
ProfileTargetGain=T_PowerGain.tpowGain(ProfileTarget,np.arange(7000),1.2)
ProfileTargetGain=DataNormalized.DataNormalized(ProfileTargetGain)/255 
pyplot.imshow(ProfileTargetGain,vmin=np.min(ProfileTargetGain),vmax=np.max(ProfileTargetGain),extent=(0,5,5,0))
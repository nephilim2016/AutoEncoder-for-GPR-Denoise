#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:06:37 2020

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot,cm
import T_PowerGain
import skimage.transform


# data=np.load('./TunnelLining.npy')
# ProfileTargetGain=T_PowerGain.tpowGain(data,np.arange(7000)/4,0.9)
# ProfileTargetGain=skimage.transform.resize(ProfileTargetGain,(7000,1000),mode='symmetric')
# pyplot.imshow(ProfileTargetGain,vmin=0.8*np.min(ProfileTargetGain),vmax=0.8*np.max(ProfileTargetGain),extent=(0,1,0.618,0),cmap=cm.seismic)
# ax=pyplot.gca()
# ax.set_xticks(np.linspace(0,1,6))
# ax.set_xticklabels([0,1,2,3,4,5])
# ax.set_yticks(np.linspace(0,0.618,8))
# ax.set_yticklabels([0,10,20,30,40,50,60,70])
# pyplot.xlabel('Scan (m)')
# pyplot.ylabel('Time (ns)')
# pyplot.savefig('TunnelLiningProfile.png',dpi=1000)


data=np.load('./ComplexProfileClean.npy')
ProfileTargetGain=T_PowerGain.tpowGain(data,np.arange(7000)/4,0.9)
ProfileTargetGain=skimage.transform.resize(ProfileTargetGain,(7000,1000),mode='symmetric')
pyplot.imshow(ProfileTargetGain,vmin=0.8*np.min(ProfileTargetGain),vmax=0.8*np.max(ProfileTargetGain),extent=(0,1,0.618,0),cmap=cm.seismic)
ax=pyplot.gca()
ax.set_xticks(np.linspace(0,1,6))
ax.set_xticklabels([0,1,2,3,4,5])
ax.set_yticks(np.linspace(0,0.618,8))
ax.set_yticklabels([0,10,20,30,40,50,60,70])
pyplot.xlabel('Scan (m)')
pyplot.ylabel('Time (ns)')
pyplot.savefig('ComplexCleanProfile.png',dpi=1000)
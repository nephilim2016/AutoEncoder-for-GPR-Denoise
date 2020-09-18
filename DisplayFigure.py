#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 19:42:41 2020

@author: nephilim
"""
import numpy as np
import my_Im2col
from matplotlib import pyplot,cm

def GetPatch(Image,patch_size,slidingDis):
    blocks,idx=my_Im2col.my_im2col(Image,patch_size,slidingDis)
    return blocks,idx

def PlotResultFigure(ProfileClean,ProfileTargetGain,Mode,patch_size,slidingDis,key): 
    # Make Patches of Target Profile
    Patch,Patch_Idx=GetPatch(ProfileTargetGain,patch_size,slidingDis)
    data_=np.zeros((Patch.shape[1],patch_size[0],patch_size[1],1))
    for idx in range(Patch.shape[1]):
        data_[idx,:,:,0]=Patch[:,idx].reshape(patch_size)
    # Predict data from autoencoder
    x_decoded=Mode.predict(data_)
    x_decoded=x_decoded[:,:,:,0]
    data=np.zeros((int(np.prod(patch_size)),x_decoded.shape[0]))
    for idx in range(x_decoded.shape[0]):
        data[:,idx]=x_decoded[idx,:,:].ravel()    
    # Collect Patch
    rows,cols=my_Im2col.ind2sub(np.array(ProfileTargetGain.shape)-patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(ProfileTargetGain.shape)
    Weight=np.zeros(ProfileTargetGain.shape)
    count=0
    for index in range(len(cols)):
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:]
        IMout[row:row+patch_size[0],col:col+patch_size[1]]+=block
        Weight[row:row+patch_size[0],col:col+patch_size[1]]+=np.ones(patch_size)
        count+=1
    NewData=IMout/Weight
    
    if key:
        # Plot Figure
        pyplot.figure()
        pyplot.imshow(NewData,vmin=np.min(NewData),vmax=1.1*np.max(NewData),cmap=cm.seismic,extent=(0,5,5,0))
        ax=pyplot.gca()
        ax.set_yticks(np.linspace(0,5,6))
        ax.set_yticklabels([0,7.50,15,22.5,30,37.5])
        pyplot.xlabel('x/m')
        pyplot.ylabel('t/ns')
        # pyplot.savefig('CNNDataDenoise600dpi.svg',dpi=600)
        # pyplot.savefig('CNNDataDenoise600dpi.png',dpi=600)
        # pyplot.savefig('CNNDataDenoise1000dpi.svg',dpi=1000)
        # pyplot.savefig('CNNDataDenoise1000dpi.png',dpi=1000)
    
        pyplot.figure()
        pyplot.imshow(ProfileTargetGain,vmin=np.min(ProfileTargetGain),vmax=np.max(ProfileTargetGain),cmap=cm.seismic,extent=(0,5,5,0))
        ax=pyplot.gca()
        ax.set_yticks(np.linspace(0,5,6))
        ax.set_yticklabels([0,7.50,15,22.5,30,37.5])
        pyplot.xlabel('x/m')
        pyplot.ylabel('t/ns')
        # pyplot.savefig('CNNStochasticDenoise600dpi.svg',dpi=600)
        # pyplot.savefig('CNNStochasticDenoise600dpi.png',dpi=600)
        # pyplot.savefig('CNNStochasticDenoise1000dpi.svg',dpi=1000)
        # pyplot.savefig('CNNStochasticDenoise1000dpi.png',dpi=1000)
        
        # pyplot.imshow(ProfileGainNoise,cmap=cm.seismic)
        
        pyplot.figure()
        pyplot.imshow(ProfileClean,vmin=np.min(ProfileClean),vmax=np.max(ProfileClean),cmap=cm.seismic,extent=(0,5,5,0))
        ax=pyplot.gca()
        ax.set_yticks(np.linspace(0,5,6))
        ax.set_yticklabels([0,7.50,15,22.5,30,37.5])
        pyplot.xlabel('x/m')
        pyplot.ylabel('t/ns')
    
    return NewData
    
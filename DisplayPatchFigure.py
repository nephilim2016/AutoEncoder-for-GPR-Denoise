#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 20:45:29 2020

@author: nephilim
"""
import numpy as np
import my_Im2col
from matplotlib import pyplot,cm
import skimage.transform
import T_PowerGain
import DataNormalized


def GetPatch(Image,patch_size,slidingDis):
    blocks,idx=my_Im2col.my_im2col(Image,patch_size,slidingDis)
    return blocks,idx

def GetPatchData(ProfileTarget,patch_size,slidingDis):
    Patch,Patch_Idx=GetPatch(ProfileTarget,patch_size,slidingDis)
    data=np.zeros((Patch.shape[1],patch_size[0],patch_size[1]))
    for idx in range(Patch.shape[1]):
        data[idx,:,:]=Patch[:,idx].reshape(patch_size)
    return data

def DisplayPatch(Patch,PatchNoise,numRows,numCols):
    bb=2
    SizeForEachImage=34
    I=np.ones((SizeForEachImage*numRows+bb,SizeForEachImage*numCols+bb))*(-1e6)
    INoise=np.ones((SizeForEachImage*numRows+bb,SizeForEachImage*numCols+bb))*(-1e6)
    maxRandom=Patch.shape[0]
    index=np.random.randint(0,maxRandom,size=maxRandom)
    counter=0
    for j in range(numRows):
        for i in range(numCols):
            I[bb+j*SizeForEachImage:(j+1)*SizeForEachImage+bb-2,bb+i*SizeForEachImage:(i+1)*SizeForEachImage+bb-2]=Patch[index[counter]]
            INoise[bb+j*SizeForEachImage:(j+1)*SizeForEachImage+bb-2,bb+i*SizeForEachImage:(i+1)*SizeForEachImage+bb-2]=PatchNoise[index[counter]]
            counter+=1
    # I-=np.min(I)
    # I/=np.max(I)
    # INoise-=np.min(INoise)
    # INoise/=np.max(INoise)
    return I,INoise
    
    

if __name__=='__main__':
    patch_size=(32,32)
    slidingDis=4
    ProfileTarget=np.load('./GPR_Modelling/ProfileAutoEncoder/%s_iter_record_%s_comp.npy'%(48,0))
    # ProfileTarget=np.load('./ProfileClean.npy')
    
    
    ProfileTarget=T_PowerGain.tpowGain(ProfileTarget,np.arange(7000),0.8)
    ProfileTarget=skimage.transform.resize(ProfileTarget,(256,256),mode='edge')
    ProfileTarget=DataNormalized.DataNormalized(ProfileTarget)/255
    
    ProfileTarget_=np.load('./GPR_Modelling/ProfileAutoEncoder/%s_iter_record_%s_comp.npy'%(48,0))
    ProfileTargetGain=T_PowerGain.tpowGain(ProfileTarget_,np.arange(7000),0.8)
    ProfileTargetGain=skimage.transform.resize(ProfileTargetGain,(256,256),mode='edge')
    ProfileNoiseTarget=ProfileTargetGain+2e4*(np.random.random(ProfileTargetGain.shape)-0.5)
    ProfileNoiseTarget=DataNormalized.DataNormalized(ProfileNoiseTarget)/255 
    
    
    Patch=GetPatchData(ProfileTarget,patch_size,slidingDis)
    PatchNoise=GetPatchData(ProfileNoiseTarget,patch_size,slidingDis)
    
    I,INoise=DisplayPatch(Patch,PatchNoise,16,16)
    pyplot.figure()
    # pyplot.subplot(1,2,1)
    pyplot.imshow(I,vmax=np.max(Patch),vmin=np.min(Patch),cmap=cm.seismic)
    pyplot.axis('off')
    pyplot.savefig('Patch0.png',dpi=1000)
    
    pyplot.figure()
    # pyplot.subplot(1,2,2)
    pyplot.imshow(INoise,vmax=np.max(PatchNoise),vmin=np.min(PatchNoise),cmap=cm.seismic)
    pyplot.axis('off')
    pyplot.savefig('Patch1.png',dpi=1000)
    
    
    
    
    
    ProfileTarget=np.load('./GPR_Modelling/ProfileAutoEncoder/%s_iter_record_%s_comp.npy'%(48,0))    
    ProfileTarget=T_PowerGain.tpowGain(ProfileTarget,np.arange(7000),0.8)
    ProfileTarget=skimage.transform.resize(ProfileTarget,(256,256),mode='edge')
    
    
    ProfileNoiseTarget=ProfileTarget+2e4*(np.random.random(ProfileTarget.shape)-0.5)
    
    pyplot.figure()
    # pyplot.subplot(1,2,2)
    pyplot.imshow(ProfileTarget,vmax=np.max(ProfileTarget),vmin=np.min(ProfileTarget),cmap=cm.seismic)
    pyplot.axis('off')
    pyplot.savefig('Original.png',dpi=1000)
    
    
    
    pyplot.figure()
    # pyplot.subplot(1,2,2)
    pyplot.imshow(ProfileNoiseTarget,vmax=np.max(ProfileNoiseTarget),vmin=np.min(ProfileNoiseTarget),cmap=cm.seismic)
    pyplot.axis('off')
    pyplot.savefig('Noise.png',dpi=1000)
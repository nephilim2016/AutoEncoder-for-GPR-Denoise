#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:16:31 2020

@author: nephilim
"""

import numpy as np
import my_Im2col
from matplotlib import pyplot,cm
import skimage.transform
import T_PowerGain
import DataNormalized
import Build_AutoEncoder
import Reshape2Encoder

def GetPatch(Image,patch_size,slidingDis):
    blocks,idx=my_Im2col.my_im2col(Image,patch_size,slidingDis)
    return blocks,idx

def CalculationSNR(Image,Noise):
    frac_up=np.sum(Image**2)
    frac_down=np.sum((Image-Noise)**2)
    SNR=10*np.log10(frac_up/frac_down)
    return SNR

if __name__=='__main__':
    # Make Target Profile Patch
    ProfileTarget=np.load('./GPR_Modelling/ProfileAutoEncoder/%s_iter_record_%s_comp.npy'%(99,1))
    ProfileTargetGain=T_PowerGain.tpowGain(ProfileTarget,np.arange(7000)/4,0.9)
    ProfileTargetGain=skimage.transform.resize(ProfileTargetGain,(256,256),mode='edge')
    ProfileTargetGain=ProfileTargetGain+2e4*(np.random.random(ProfileTargetGain.shape)-0.5)
    ProfileTargetGain=DataNormalized.DataNormalized(ProfileTargetGain)/255
    
    patch_size=(32,32)
    slidingDis=4
    Patch,Patch_Idx=GetPatch(ProfileTargetGain,patch_size,slidingDis)
    
    # Make Dataset of Profile without Noise
    Iteration=100
    compare=0
    ProfileGain_train=[]
    for iteration in range(Iteration):
        Profile=np.load('./GPR_Modelling/ProfileAutoEncoder/%s_iter_record_%s_comp.npy'%(iteration,compare))
        ProfileGain=T_PowerGain.tpowGain(Profile,np.arange(7000)/4,0.9)
        ProfileGain=skimage.transform.resize(ProfileGain,(256,256),mode='edge')
        ProfileGain=DataNormalized.DataNormalized(ProfileGain)/255
        ProfileGain_Patch,_=GetPatch(ProfileGain,patch_size,slidingDis)
        ProfileGain_train.append(ProfileGain_Patch)
    ProfileGain_train=np.array(ProfileGain_train)
    Profile_train=Reshape2Encoder.ReshapeData2Encoder(ProfileGain_train,patch_size)
    del ProfileGain,ProfileGain_Patch,ProfileGain_train 
    
    # Make Dataset of Profile with Noise
    Iteration=100
    compare=1
    ProfileNoiseGain_train=[]
    for iteration in range(Iteration):
        Profile=np.load('./GPR_Modelling/ProfileAutoEncoder/%s_iter_record_%s_comp.npy'%(iteration,compare))
        ProfileNoiseGain=T_PowerGain.tpowGain(Profile,np.arange(7000)/4,0.9)
        ProfileNoiseGain=skimage.transform.resize(ProfileNoiseGain,(256,256),mode='edge')
        ProfileNoiseGain=ProfileNoiseGain+2e4*(np.random.random(ProfileNoiseGain.shape)-0.5)
        ProfileNoiseGain=DataNormalized.DataNormalized(ProfileNoiseGain)/255
        ProfileNoiseGain_Patch,_=GetPatch(ProfileNoiseGain,patch_size,slidingDis)
        ProfileNoiseGain_train.append(ProfileNoiseGain_Patch)
    ProfileNoiseGain_train=np.array(ProfileNoiseGain_train)
    ProfileNoise_train=Reshape2Encoder.ReshapeData2Encoder(ProfileNoiseGain_train,patch_size)    
    del ProfileNoiseGain,ProfileNoiseGain_Patch,ProfileNoiseGain_train 
    # Build Autoencoer
    AutoEncoder=Build_AutoEncoder.BuildAutoEncoder(ImageShape=tuple(list(patch_size)+[1,]),filters=[16,32,64,128],kernel_size=[3,3,3,3],latent_dim=256)
    inputs_train=ProfileNoise_train[:-2000]
    outputs_train=Profile_train[:-2000]
    inputs_validation=ProfileNoise_train[-2000:None]
    outputs_validation=Profile_train[-2000:None]
    epochs=10
    history,test_loss,Mode=Build_AutoEncoder.AutoEncoderTraining(AutoEncoder,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation)
    
    history_dict=history.history
    history_dict.keys()
    loss=history_dict['loss']
    
    data_=np.zeros((Patch.shape[1],patch_size[0],patch_size[1],1))
    for idx in range(Patch.shape[1]):
        data_[idx,:,:,0]=Patch[:,idx].reshape(patch_size)
    x_decoded=Mode.predict(data_)
    x_decoded=x_decoded[:,:,:,0]
    data=np.zeros((int(np.prod(patch_size)),x_decoded.shape[0]))
    for idx in range(x_decoded.shape[0]):
        data[:,idx]=x_decoded[idx,:,:].ravel()    
    
    
    
    rows,cols=my_Im2col.ind2sub(np.array(ProfileTargetGain.shape)-patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(ProfileTargetGain.shape)
    Weight=np.zeros(ProfileTargetGain.shape)
    count=0
    for index in range(len(cols)):
        
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:]
        # print(block.shape)
        # print(IMout[row:row+28,col:col+28].shape)
        IMout[row:row+patch_size[0],col:col+patch_size[1]]+=block
        Weight[row:row+patch_size[0],col:col+patch_size[1]]+=np.ones(patch_size)
        count+=1
        
    # NewData=(ProfileTarget+0.8*IMout)/(1+0.8*Weight)
    NewData=IMout/Weight
    # SNR=CalculationSNR(ProfileGain,NewData)
    # print(SNR)
    
    pyplot.figure()
    # pyplot.imshow(NewData,vmin=0.5*np.min(NewData),vmax=np.max(NewData),cmap=cm.seismic)
    pyplot.imshow(NewData,vmin=np.min(NewData),vmax=np.max(NewData),extent=(0,5,5,0))
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
    pyplot.imshow(ProfileTargetGain,vmin=np.min(ProfileTargetGain),vmax=np.max(ProfileTargetGain),extent=(0,5,5,0))
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
    ProfileTarget=np.load('./GPR_Modelling/ProfileAutoEncoder/%s_iter_record_%s_comp.npy'%(99,0))
    ProfileTarget=T_PowerGain.tpowGain(ProfileTarget,np.arange(7000)/4,0.9)
    ProfileTarget=skimage.transform.resize(ProfileTarget,(256,256),mode='edge')
    
    ProfileTarget=DataNormalized.DataNormalized(ProfileTarget)/255
    pyplot.figure()
    pyplot.imshow(ProfileTarget,vmin=np.min(ProfileTarget),vmax=np.max(ProfileTarget),extent=(0,5,5,0))
    ax=pyplot.gca()
    ax.set_yticks(np.linspace(0,5,6))
    ax.set_yticklabels([0,7.50,15,22.5,30,37.5])
    pyplot.xlabel('x/m')
    pyplot.ylabel('t/ns')
    
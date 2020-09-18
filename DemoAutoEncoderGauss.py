#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:33:58 2020

@author: nephilim
"""

import numpy as np
import my_Im2col
import skimage.transform
import T_PowerGain
import DataNormalized
import Build_AutoEncoder
import Reshape2Encoder
import DisplayFigure
from matplotlib import pyplot,cm
import json


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
    patch_size=(32,32)
    slidingDis=4
    # Make Dataset of Profile without Noise
    Iteration=100
    compare=1
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
    # AutoEncoder=Build_AutoEncoder.BuildAutoEncoder(ImageShape=tuple(list(patch_size)+[1,]),filters=[16,32,64],kernel_size=[5,5,5],latent_dim=256)
    # AutoEncoder=Build_AutoEncoder.BuildAtrousAutoEncoder(ImageShape=tuple(list(patch_size)+[1,]),filters=[16,32,64],kernel_size=[5,5,5],latent_dim=256)
    AutoEncoder=Build_AutoEncoder.BuildDropOutAutoEncoder(ImageShape=tuple(list(patch_size)+[1,]),filters=[16,32,64],kernel_size=[3,3,3],latent_dim=256)
    # AutoEncoder=Build_AutoEncoder.BuildResidualConnectionAutoEncoder(ImageShape=tuple(list(patch_size)+[1,]),filters=[16,32,64],kernel_size=[5,5,5],latent_dim=256)

    
    inputs_train=ProfileNoise_train[:-2000]
    outputs_train=Profile_train[:-2000]
    inputs_validation=ProfileNoise_train[-2000:None]
    outputs_validation=Profile_train[-2000:None]
    epochs=10
    save_path_name='./GaussStochasticNormalAutoEncoder400MB_iteration10_monitor_DropOutkernel3_windows3_[16,32,64]'
    history,test_loss,Mode=Build_AutoEncoder.AutoEncoderTraining(AutoEncoder,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name)
    # history,test_loss,Mode=Build_AutoEncoder.AtrousAutoEncoderTraining(AutoEncoder,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name)

    # history,test_loss,Mode=Build_AutoEncoder.DropOutAutoEncoderTraining(AutoEncoder,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name)
    # history,test_loss,Mode=Build_AutoEncoder.ResidualConnectionAutoEncoderTraining(AutoEncoder,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name)

    # Mode.save('GaussNormalAutoEncoder400MB_10iteration.h5')
    with open(save_path_name+'.json', 'w') as f:
        json.dump(history.history,f)
    
    history_dict=history.history
    history_dict.keys()
    loss=history_dict['loss']
    val_loss=history_dict['val_loss']
    epochs_axis=range(1,epochs+1)
    
    pyplot.figure()
    pyplot.plot(epochs_axis,loss,'bo',label='Training loss')
    pyplot.plot(epochs_axis,val_loss,'b+',label='Validation loss')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Loss')
    pyplot.legend()
    
    # import keras
    # Mode=keras.models.load_model('GaussNormalAutoEncoder400MB_10iteration_monitor_kernel_3.h5')
    # Read Target Profile
    ProfileTarget=np.load('./TunnelLining.npy')
    ProfileTargetGain=T_PowerGain.tpowGain(ProfileTarget,np.arange(7000)/4,0.9)
    ProfileTargetGain=skimage.transform.resize(ProfileTargetGain,(256,256),mode='edge')
    ProfileTargetGain=ProfileTargetGain+2e4*(np.random.random(ProfileTargetGain.shape)-0.5)
    ProfileTargetGain=DataNormalized.DataNormalized(ProfileTargetGain)/255
    # pyplot.imshow(ProfileTargetGain,vmin=np.min(ProfileTargetGain),vmax=np.max(ProfileTargetGain),extent=(0,5,5,0))
    # Read Profile without Noise
    ProfileClean=np.load('./TunnelLining.npy')
    ProfileClean=T_PowerGain.tpowGain(ProfileClean,np.arange(7000)/4,0.9)
    ProfileClean=skimage.transform.resize(ProfileClean,(256,256),mode='edge')
    ProfileClean=DataNormalized.DataNormalized(ProfileClean)/255
    
    DisplayFigure.PlotResultFigure(ProfileClean,ProfileTargetGain,Mode,patch_size,slidingDis)
    
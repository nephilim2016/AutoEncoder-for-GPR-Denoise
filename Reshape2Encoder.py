#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:00:42 2020

@author: nephilim
"""
import numpy as np

def ReshapeData2Encoder(ProfileGain_train,patch_size):
    ProfileGain_train_=np.zeros((ProfileGain_train.shape[1],ProfileGain_train.shape[0]*ProfileGain_train.shape[2]))
    for idx in range(ProfileGain_train.shape[0]):
        ProfileGain_train_[:,idx*ProfileGain_train.shape[2]:idx*ProfileGain_train.shape[2]+ProfileGain_train.shape[2]]=ProfileGain_train[idx,:,:]
    
    Profile_train=np.zeros((ProfileGain_train_.shape[1],)+patch_size)
    for idx in range(ProfileGain_train_.shape[1]):
        Profile_train[idx,:,:]=ProfileGain_train_[:,idx].reshape(patch_size)
    Profile_train_=Profile_train.reshape((-1,)+patch_size+(1,))
    
    return Profile_train_
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:58:10 2020

@author: nephilim
"""
import keras
import numpy as np
import my_Im2col
from matplotlib import pyplot,cm
import skimage.transform
import T_PowerGain
import DataNormalized
import Reshape2Encoder


def GetPatch(Image,patch_size,slidingDis):
    blocks,idx=my_Im2col.my_im2col(Image,patch_size,slidingDis)
    return blocks,idx

def CalculationSNR(Image,Noise):
    frac_up=np.sum(Image**2)
    frac_down=np.sum((Image-Noise)**2)
    SNR=10*np.log10(frac_up/frac_down)
    return SNR

class AutoEncoder():
    def __init__(self,ImageShape,filters,kernel_size,latent_dim):
        self.ImageShape=ImageShape
        self.filters=filters
        self.kernel_size=kernel_size
        self.latent_dim=latent_dim
    
    def Encoder(self):
        self.Encoder_Input=keras.Input(shape=self.ImageShape,name='Encoder_Input_2D')
        x=self.Encoder_Input
        for idx,_ in enumerate(self.filters):
            x=keras.layers.Conv2D(filters=self.filters[idx],kernel_size=self.kernel_size[idx],activation='relu',padding='same')(x)
            x=keras.layers.BatchNormalization()(x)
            x=keras.layers.MaxPool2D((2,2))(x)
                        
        self.shape=keras.backend.int_shape(x)
        # print(self.shape)
        x=keras.layers.Flatten()(x)
        Encoder_Output=keras.layers.Dense(self.latent_dim,name='Encoder_Ouput_1D')(x)
        self.EncoderMode=keras.models.Model(inputs=self.Encoder_Input,outputs=Encoder_Output,name='EncoderPart')
        self.EncoderMode.summary()        
        self.EncoderMode.compile(loss='mse',optimizer='adam')

    def Decoder(self):
        Decoder_Input=keras.Input(shape=(self.latent_dim,),name='Decoder_Input_1D')
        x=keras.layers.Dense(self.shape[1]*self.shape[2]*self.shape[3])(Decoder_Input)
        x=keras.layers.Reshape((self.shape[1],self.shape[2],self.shape[3]))(x)
        for idx,_ in enumerate(self.filters):
            x=keras.layers.Conv2DTranspose(filters=self.filters[len(self.filters)-idx-1],kernel_size=self.kernel_size[len(self.kernel_size)-idx-1],activation='relu',padding='same')(x)
            x=keras.layers.BatchNormalization()(x)
            x=keras.layers.UpSampling2D((2,2))(x)
        Decoder_Output=keras.layers.Conv2DTranspose(filters=1,kernel_size=5,activation='sigmoid',padding='same',name='Decoder_Output_1D')(x)
        self.DecoderMode=keras.models.Model(inputs=Decoder_Input,outputs=Decoder_Output)
        self.DecoderMode.summary()  
        self.DecoderMode.compile(loss='mse',optimizer='adam')



if __name__=='__main__':
    AutoEncoder_=AutoEncoder(ImageShape=(512,512,1),filters=[16,32,64],kernel_size=[5,5,5],latent_dim=256)
    AutoEncoder_.Encoder()
    
    
    patch_size=(512,512)
    slidingDis=64
    Iteration=100
    ProfileGain_train=[]
    for iteration in range(Iteration):
        Profile=np.load('./GPR_Modelling/Profile_TunnelLining/TunnelLining_Iter_%s.npy'%iteration)
        ProfileGain=T_PowerGain.tpowGain(Profile,np.arange(7000)/4,0.9)
        ProfileGain=skimage.transform.resize(ProfileGain,(512,512),mode='edge')
        ProfileGain=DataNormalized.DataNormalized(ProfileGain)/255
        ProfileGain_Patch,_=GetPatch(ProfileGain,patch_size,slidingDis)
        ProfileGain_train.append(ProfileGain_Patch)
        
    Iteration=100
    compare=1
    for iteration in range(Iteration):
        Profile=np.load('./GPR_Modelling/ProfileAutoEncoder/%s_iter_record_%s_comp.npy'%(iteration,compare))
        ProfileGain=T_PowerGain.tpowGain(Profile,np.arange(7000)/4,0.9)
        ProfileGain=skimage.transform.resize(ProfileGain,(512,512),mode='edge')
        ProfileGain=DataNormalized.DataNormalized(ProfileGain)/255
        ProfileGain_Patch,_=GetPatch(ProfileGain,patch_size,slidingDis)
        ProfileGain_train.append(ProfileGain_Patch)
        
    ProfileGain_train=np.array(ProfileGain_train)
    Profile_train=Reshape2Encoder.ReshapeData2Encoder(ProfileGain_train,patch_size)
    del ProfileGain,ProfileGain_Patch,ProfileGain_train 
    
    
    layer_outputs=[layer.output for layer in AutoEncoder_.EncoderMode.layers[1:None]]
    activation_model=keras.models.Model(inputs=AutoEncoder_.EncoderMode.input,outputs=layer_outputs)
    activations=activation_model.predict(Profile_train[2:3,:,:])
    
    # pyplot.figure()
    # pyplot.imshow(ProfileGain)
    # pyplot.figure()
    # pyplot.imshow(activations[1][0,:,:,0])
    # pyplot.figure()
    # pyplot.imshow(activations[3][0,:,:,0])
    # pyplot.figure()
    # pyplot.imshow(activations[5][0,:,:,0])
    
    
    # pyplot.figure()
    # pyplot.imshow(ProfileGain)
    # pyplot.figure()
    # pyplot.imshow(display_grid[0,:,:])
    # pyplot.figure()
    # pyplot.imshow(display_grid[1,:,:])
    # pyplot.figure()
    # pyplot.imshow(display_grid[2,:,:])
    # pyplot.figure()
    # pyplot.imshow(Profile_train[1541,:,:,0])
    # # pyplot.figure()
    # # pyplot.imshow(activations[1][0,:,:,0])
    # # pyplot.figure()
    # # pyplot.imshow(activations[3][0,:,:,0])
    # # pyplot.figure()
    # # pyplot.imshow(activations[5][0,:,:,0])

    # image_per_row=16
    # for layer_activation in activations:
    #     n_features=layer_activation.shape[-1]
    #     size=layer_activation.shape[1]
    #     n_cols=n_features//image_per_row
    #     display_grid=np.zeros((size*n_cols,image_per_row*size))
    #     for col in range(n_cols):
    #         for row in range(image_per_row):
    #             channel_image=layer_activation[0,:,:,col*image_per_row+row]
    #             display_grid[col*size:(col+1)*size,row*size:(row+1)*size]=channel_image
    #     pyplot.figure()
    #     pyplot.imshow(display_grid[:,:])
    #     pyplot.axis('off')
    pyplot.figure()
    data=Profile_train[2:3,:,:]
    data=data[0,:,:,0]
    pyplot.imshow(data,cmap=cm.seismic)
    pyplot.axis('off')
    pyplot.savefig('OriginalInput.png',dpi=1000)
    
    FirstConv2DLayer=activations[2]
    image_per_row=4
    n_features=FirstConv2DLayer.shape[-1]
    size=FirstConv2DLayer.shape[1]
    n_cols=n_features//image_per_row
    display_grid=np.zeros((n_cols*size,image_per_row*size))
    for col in range(n_cols):
        for row in range(image_per_row):
            channel_image=FirstConv2DLayer[0,:,:,col*image_per_row+row]
            display_grid[col*size:(col+1)*size,row*size:(row+1)*size]=channel_image
    pyplot.figure()
    pyplot.imshow(display_grid[:,:],vmin=-0.2,vmax=0.4,cmap=cm.seismic)
    pyplot.axis('off')
    pyplot.savefig('FirstConv2D.png',dpi=1000)
    
    FirstConv2DLayer=activations[5]
    image_per_row=8
    n_features=FirstConv2DLayer.shape[-1]
    size=FirstConv2DLayer.shape[1]
    n_cols=n_features//image_per_row
    display_grid=np.zeros((n_cols*size,image_per_row*size))
    for col in range(n_cols):
        for row in range(image_per_row):
            channel_image=FirstConv2DLayer[0,:,:,col*image_per_row+row]
            display_grid[col*size:(col+1)*size,row*size:(row+1)*size]=channel_image
    pyplot.figure()
    pyplot.imshow(display_grid[:,:],vmin=-0.2,vmax=0.4,cmap=cm.seismic)
    pyplot.axis('off')
    pyplot.savefig('SecondConv2D.png',dpi=1000)
    
    FirstConv2DLayer=activations[8]
    image_per_row=8
    n_features=FirstConv2DLayer.shape[-1]
    size=FirstConv2DLayer.shape[1]
    n_cols=n_features//image_per_row
    display_grid=np.zeros((n_cols*size,image_per_row*size))
    for col in range(n_cols):
        for row in range(image_per_row):
            channel_image=FirstConv2DLayer[0,:,:,col*image_per_row+row]
            display_grid[col*size:(col+1)*size,row*size:(row+1)*size]=channel_image
    pyplot.figure()
    pyplot.imshow(display_grid[:,:],vmin=-0.2,vmax=0.4,cmap=cm.seismic)
    pyplot.axis('off')
    pyplot.savefig('ThirdConv2D.png',dpi=1000)
    
    
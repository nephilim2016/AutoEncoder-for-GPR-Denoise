#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:58:10 2020

@author: nephilim
"""
import keras

# def mse(y_true,y_pred):
#     return keras.backend.mean(keras.backend.square(y_pred-y_true),axis=-1)

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
        Decoder_Output=keras.layers.Conv2DTranspose(filters=1,kernel_size=3,activation='sigmoid',padding='same',name='Decoder_Output_1D')(x)
        self.DecoderMode=keras.models.Model(inputs=Decoder_Input,outputs=Decoder_Output)
        self.DecoderMode.summary()  
        self.DecoderMode.compile(loss='mse',optimizer='adam')
        
    def DropOutEncoder(self):
        self.Encoder_Input=keras.Input(shape=self.ImageShape,name='Encoder_Input_2D')
        x=self.Encoder_Input
        for idx,_ in enumerate(self.filters):
            x=keras.layers.Conv2D(filters=self.filters[idx],kernel_size=self.kernel_size[idx],activation='relu',padding='same')(x)
            x=keras.layers.BatchNormalization()(x)
            x=keras.layers.MaxPool2D((2,2))(x)
            x=keras.layers.Dropout(0.2)(x)
            # if idx==1:
                # residual=keras.layers.Conv2D(filters=self.filters[idx],kernel_size=3,padding='same')(self.Encoder_Input)
                # residual=keras.layers.MaxPool2D((4,4))(residual)
                # x=keras.layers.add([x,residual])
        # residual=keras.layers.Conv2D(filters=self.filters[-1],kernel_size=3,strides=2**len(self.filters),padding='same')(self.Encoder_Input)
        # x=keras.layers.add([x,residual])
        self.shape=keras.backend.int_shape(x)
        # print(self.shape)
        x=keras.layers.Flatten()(x)
        Encoder_Output=keras.layers.Dense(self.latent_dim,name='Encoder_Ouput_1D')(x)
        self.EncoderMode=keras.models.Model(inputs=self.Encoder_Input,outputs=Encoder_Output,name='EncoderPart')
        self.EncoderMode.summary()        
        self.EncoderMode.compile(loss='mse',optimizer='adam')

    def DropOutDecoder(self):
        Decoder_Input=keras.Input(shape=(self.latent_dim,),name='Decoder_Input_1D')
        x=keras.layers.Dense(self.shape[1]*self.shape[2]*self.shape[3])(Decoder_Input)
        x=keras.layers.Reshape((self.shape[1],self.shape[2],self.shape[3]))(x)
        for idx,_ in enumerate(self.filters):
            x=keras.layers.Conv2DTranspose(filters=self.filters[len(self.filters)-idx-1],kernel_size=self.kernel_size[len(self.kernel_size)-idx-1],activation='relu',padding='same')(x)
            x=keras.layers.BatchNormalization()(x)
            x=keras.layers.UpSampling2D((2,2))(x)
            x=keras.layers.Dropout(0.2)(x)
        Decoder_Output=keras.layers.Conv2DTranspose(filters=1,kernel_size=3,activation='sigmoid',padding='same',name='Decoder_Output_1D')(x)
        self.DecoderMode=keras.models.Model(inputs=Decoder_Input,outputs=Decoder_Output)
        self.DecoderMode.summary()  
        self.DecoderMode.compile(loss='mse',optimizer='adam')
        
        
    def ResidualConnectionEncoder(self):
        self.Encoder_Input=keras.Input(shape=self.ImageShape,name='Encoder_Input_2D')
        x=self.Encoder_Input
        for idx,_ in enumerate(self.filters):
            x=keras.layers.Conv2D(filters=self.filters[idx],kernel_size=self.kernel_size[idx],activation='relu',padding='same')(x)
            x=keras.layers.BatchNormalization()(x)
            x=keras.layers.MaxPool2D((2,2))(x)
            x=keras.layers.Dropout(0.2)(x)
            if idx==0:
                residual=keras.layers.Conv2D(filters=self.filters[idx],kernel_size=5,padding='same')(self.Encoder_Input)
                residual=keras.layers.BatchNormalization()(residual)
                residual=keras.layers.MaxPool2D((2,2))(residual)
                x=keras.layers.add([x,residual])
            if idx==1:
                residual=keras.layers.Conv2D(filters=self.filters[idx],kernel_size=5,padding='same')(self.Encoder_Input)
                residual=keras.layers.BatchNormalization()(residual)
                residual=keras.layers.MaxPool2D((4,4))(residual)
                x=keras.layers.add([x,residual])
        # residual=keras.layers.Conv2D(filters=self.filters[-1],kernel_size=3,strides=2**len(self.filters),padding='same')(self.Encoder_Input)
        # x=keras.layers.add([x,residual])
        self.shape=keras.backend.int_shape(x)
        # print(self.shape)
        x=keras.layers.Flatten()(x)
        Encoder_Output=keras.layers.Dense(self.latent_dim,name='Encoder_Ouput_1D')(x)
        self.EncoderMode=keras.models.Model(inputs=self.Encoder_Input,outputs=Encoder_Output,name='EncoderPart')
        self.EncoderMode.summary()        
        self.EncoderMode.compile(loss='mse',optimizer='adam')

    def ResidualConnectionDecoder(self):
        Decoder_Input=keras.Input(shape=(self.latent_dim,),name='Decoder_Input_1D')
        x=keras.layers.Dense(self.shape[1]*self.shape[2]*self.shape[3])(Decoder_Input)
        x=keras.layers.Reshape((self.shape[1],self.shape[2],self.shape[3]))(x)
        for idx,_ in enumerate(self.filters):
            x=keras.layers.Conv2DTranspose(filters=self.filters[len(self.filters)-idx-1],kernel_size=self.kernel_size[len(self.kernel_size)-idx-1],activation='relu',padding='same')(x)
            x=keras.layers.BatchNormalization()(x)
            x=keras.layers.UpSampling2D((2,2))(x)
    
        Decoder_Output=keras.layers.Conv2DTranspose(filters=1,kernel_size=3,activation='sigmoid',padding='same',name='Decoder_Output_1D')(x)
        self.DecoderMode=keras.models.Model(inputs=Decoder_Input,outputs=Decoder_Output)
        self.DecoderMode.summary()  
        self.DecoderMode.compile(loss='mse',optimizer='adam')
        
        
    def AtrousEncoder(self):
        self.Encoder_Input=keras.Input(shape=self.ImageShape,name='Encoder_Input_2D')
        x=self.Encoder_Input
        for idx,_ in enumerate(self.filters):
            x=keras.layers.Conv2D(filters=self.filters[idx],kernel_size=self.kernel_size[idx],activation='relu',padding='same',dilation_rate=idx+1)(x)
            x=keras.layers.BatchNormalization()(x)
            x=keras.layers.MaxPool2D((2,2))(x)
            x=keras.layers.BatchNormalization()(x)
            # if idx==1:
                # residual=keras.layers.Conv2D(filters=self.filters[idx],kernel_size=3,padding='same')(self.Encoder_Input)
                # residual=keras.layers.MaxPool2D((4,4))(residual)
                # x=keras.layers.add([x,residual])
        # residual=keras.layers.Conv2D(filters=self.filters[-1],kernel_size=3,strides=2**len(self.filters),padding='same')(self.Encoder_Input)
        # x=keras.layers.add([x,residual])
        self.shape=keras.backend.int_shape(x)
        # print(self.shape)
        x=keras.layers.Flatten()(x)
        Encoder_Output=keras.layers.Dense(self.latent_dim,name='Encoder_Ouput_1D')(x)
        self.EncoderMode=keras.models.Model(inputs=self.Encoder_Input,outputs=Encoder_Output,name='EncoderPart')
        self.EncoderMode.summary()        
        self.EncoderMode.compile(loss='mse',optimizer='adam')

    def AtrousDecoder(self):
        Decoder_Input=keras.Input(shape=(self.latent_dim,),name='Decoder_Input_1D')
        x=keras.layers.Dense(self.shape[1]*self.shape[2]*self.shape[3])(Decoder_Input)
        x=keras.layers.Reshape((self.shape[1],self.shape[2],self.shape[3]))(x)
        for idx,_ in enumerate(self.filters):
            x=keras.layers.Conv2DTranspose(filters=self.filters[len(self.filters)-idx-1],kernel_size=self.kernel_size[len(self.kernel_size)-idx-1],activation='relu',padding='same',dilation_rate=len(self.kernel_size)-idx)(x)
            x=keras.layers.BatchNormalization()(x)
            x=keras.layers.UpSampling2D((2,2))(x)
            x=keras.layers.BatchNormalization()(x)
        Decoder_Output=keras.layers.Conv2DTranspose(filters=1,kernel_size=3,activation='sigmoid',padding='same',name='Decoder_Output_1D')(x)
        self.DecoderMode=keras.models.Model(inputs=Decoder_Input,outputs=Decoder_Output)
        # self.DecoderMode.summary()  
        self.DecoderMode.compile(loss='mse',optimizer='adam')
        
def BuildAutoEncoder(ImageShape=(32,32,1),filters=[32,64,128],kernel_size=[5,5,5],latent_dim=256):
    AutoEncoder_=AutoEncoder(ImageShape,filters,kernel_size,latent_dim)
    AutoEncoder_.Encoder()
    AutoEncoder_.Decoder()
    AutoEncoderMode=keras.models.Model(inputs=AutoEncoder_.Encoder_Input,outputs=AutoEncoder_.DecoderMode(AutoEncoder_.EncoderMode(AutoEncoder_.Encoder_Input)),name='AutoEncoderMode')
    AutoEncoderMode.summary()  
    AutoEncoderMode.compile(loss='mse',optimizer='adam')
    return AutoEncoderMode

def BuildDropOutAutoEncoder(ImageShape=(32,32,1),filters=[32,64,128],kernel_size=[5,5,5],latent_dim=256):
    AutoEncoder_=AutoEncoder(ImageShape,filters,kernel_size,latent_dim)
    AutoEncoder_.DropOutEncoder()
    AutoEncoder_.DropOutDecoder()
    AutoEncoderMode=keras.models.Model(inputs=AutoEncoder_.Encoder_Input,outputs=AutoEncoder_.DecoderMode(AutoEncoder_.EncoderMode(AutoEncoder_.Encoder_Input)),name='AutoEncoderMode')
    AutoEncoderMode.summary()  
    AutoEncoderMode.compile(loss='mse',optimizer='adam')
    return AutoEncoderMode

def BuildResidualConnectionAutoEncoder(ImageShape=(32,32,1),filters=[32,64,128],kernel_size=[5,5,5],latent_dim=256):
    AutoEncoder_=AutoEncoder(ImageShape,filters,kernel_size,latent_dim)
    AutoEncoder_.ResidualConnectionEncoder()
    AutoEncoder_.ResidualConnectionDecoder()
    AutoEncoderMode=keras.models.Model(inputs=AutoEncoder_.Encoder_Input,outputs=AutoEncoder_.DecoderMode(AutoEncoder_.EncoderMode(AutoEncoder_.Encoder_Input)),name='AutoEncoderMode')
    AutoEncoderMode.summary()  
    AutoEncoderMode.compile(loss='mse',optimizer='adam')
    return AutoEncoderMode

def BuildAtrousAutoEncoder(ImageShape=(32,32,1),filters=[32,64,128],kernel_size=[5,5,5],latent_dim=256):
    AutoEncoder_=AutoEncoder(ImageShape,filters,kernel_size,latent_dim)
    AutoEncoder_.AtrousEncoder()
    AutoEncoder_.AtrousDecoder()
    AutoEncoderMode=keras.models.Model(inputs=AutoEncoder_.Encoder_Input,outputs=AutoEncoder_.DecoderMode(AutoEncoder_.EncoderMode(AutoEncoder_.Encoder_Input)),name='AutoEncoderMode')
    AutoEncoderMode.summary()  
    AutoEncoderMode.compile(loss='mse',optimizer='adam')
    return AutoEncoderMode

def AutoEncoderTraining(Model,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name):
    callbacks_list=[keras.callbacks.ModelCheckpoint(filepath=save_path_name+'.h5',monitor='val_loss',save_best_only=True),\
                    keras.callbacks.TensorBoard(log_dir='./TensorBoard',histogram_freq=1,write_graph=True,write_images=True)]
    history=Model.fit(inputs_train,outputs_train,epochs=epochs,batch_size=64,callbacks=callbacks_list,validation_data=(inputs_validation,outputs_validation))
    test_loss=Model.evaluate(inputs_validation,outputs_validation)
    return history,test_loss,Model

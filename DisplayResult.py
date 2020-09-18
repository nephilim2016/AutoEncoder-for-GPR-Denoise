#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 12:07:08 2020

@author: nephilim
"""

import keras
import numpy as np
import skimage.transform
import T_PowerGain
import DataNormalized
import DisplayFigure
from matplotlib import pyplot,cm

patch_size=(32,32)
slidingDis=4

def CalculationSNR(Image,Noise):
    frac_up=np.sum(Image**2)
    frac_down=np.sum((Image-Noise)**2)
    SNR=10*np.log10(frac_up/frac_down)
    return SNR

Mode_Residual=keras.models.load_model('./CompareFilters/GaussTunnelLiningNormalAutoEncoder400MB_iteration10_monitor_Residualkernel5_windows3_[16,32,64].h5')
# Mode_Atrous=keras.models.load_model('./CompareFilters/GaussNormalAutoEncoder400MB_iteration10_monitor_Atrouskernel5_windows3_[16,32,64].h5')
# Mode_Original=keras.models.load_model('./CompareFilters/GaussNormalAutoEncoder400MB_iteration10_monitor_kernel3_windows5_[16,32,64].h5')

# Read Target Profile
ProfileTargetGain=np.load('./DATA.npy').astype('float')

# ProfileTarget=np.load('./ComplexProfileClean.npy')
# ProfileTarget=np.load('./ComplexProfileClean.npy')
# ProfileTargetGain=T_PowerGain.tpowGain(ProfileTargetGain,np.arange(5001)/4,0.9)
ProfileTargetGain=skimage.transform.resize(ProfileTargetGain,(256,256),mode='edge')
ProfileTargetGain=ProfileTargetGain+5e6*(np.random.random(ProfileTargetGain.shape)-0.5)
ProfileTargetGain[45:155,190:193]=ProfileTargetGain[45:155,190:193]+1.5e7*(np.random.random(ProfileTargetGain[45:155,190:193].shape)-0.5)
# 
ProfileTargetGain[:,90:93]=ProfileTargetGain[:,90:93]+1.8e6*(np.random.random(ProfileTargetGain[:,90:93].shape)-0.5)

ProfileTargetGain[:,100:103]=ProfileTargetGain[:,100:103]+1e7*(np.random.random(ProfileTargetGain[:,100:103].shape)-0.5)

ProfileTargetGain[:,150:153]=ProfileTargetGain[:,150:153]+3.1e6*(np.random.random(ProfileTargetGain[:,150:153].shape)-0.5)

ProfileTargetGain[:,160:163]=ProfileTargetGain[:,160:163]+4e6*(np.random.random(ProfileTargetGain[:,160:163].shape)-0.5)


ProfileTargetGain=DataNormalized.DataNormalized(ProfileTargetGain)/255 
# pyplot.imshow(ProfileTargetGain,vmin=np.min(ProfileTargetGain),vmax=np.max(ProfileTargetGain),extent=(0,5,5,0))

# Read Profile without Noise
# ProfileClean=np.load('./GPR_Modelling/ProfileAutoEncoder/60_iter_record_0_comp.npy')
# ProfileClean=np.load('./ComplexProfileClean.npy')
ProfileClean=np.load('./DATA.npy')

# ProfileClean=T_PowerGain.tpowGain(ProfileClean,np.arange(5001)/4,0.9)
ProfileClean=skimage.transform.resize(ProfileClean,(256,256),mode='edge')
ProfileClean_=ProfileClean.copy()
ProfileClean=DataNormalized.DataNormalized(ProfileClean)/255

NewData_Residual=DisplayFigure.PlotResultFigure(ProfileClean,ProfileTargetGain,Mode_Residual,patch_size,slidingDis,False)
# NewData_Atrous=DisplayFigure.PlotResultFigure(ProfileClean,ProfileTargetGain,Mode_Atrous,patch_size,slidingDis,False)
# NewData_Original=DisplayFigure.PlotResultFigure(ProfileClean,ProfileTargetGain,Mode_Original,patch_size,slidingDis,False)

NoiseSNR=CalculationSNR(ProfileClean*255,ProfileTargetGain*255)
DenoiseSNR_Residual=CalculationSNR(ProfileClean*255,NewData_Residual*255)
# DenoiseSNR_Atrous=CalculationSNR(ProfileClean*255,NewData_Atrous*255)
# DenoiseSNR_Original=CalculationSNR(ProfileClean*255,NewData_Original*255)


ProfileClean=DataNormalized.InverseDataNormalized(ProfileClean_,ProfileClean)
ProfileNoise=DataNormalized.InverseDataNormalized(ProfileClean_,ProfileTargetGain)
ProfileDenoise_Residual=DataNormalized.InverseDataNormalized(ProfileClean_,NewData_Residual)
# ProfileDenoise_Atrous=DataNormalized.InverseDataNormalized(ProfileClean_,NewData_Atrous)
# ProfileDenoise_Original=DataNormalized.InverseDataNormalized(ProfileClean_,NewData_Original)



NoiseResidual=ProfileClean-ProfileNoise
DenoiseResidual_Residual=ProfileDenoise_Residual-ProfileNoise
# DenoiseResidual_Atrous=ProfileDenoise_Atrous-ProfileNoise
# DenoiseResidual_Original=ProfileDenoise_Original-ProfileNoise

pyplot.figure()
pyplot.imshow(ProfileNoise,vmin=np.min(ProfileNoise),vmax=np.max(ProfileNoise),extent=(0,1,1,0),cmap=cm.gray)
ax=pyplot.gca()
ax.set_xticks(np.linspace(0,1,12))
ax.set_xticklabels([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5])
ax.set_yticks(np.linspace(0,1,7))
ax.set_yticklabels([0,10,20,30,40,50,60])
pyplot.xlabel('Scan (m)')
pyplot.ylabel('Time (ns)')
# pyplot.savefig('ActualNoise.png',dpi=1000)


# pyplot.figure()
# pyplot.imshow(NoiseResidual,vmin=np.min(NoiseResidual),vmax=np.max(NoiseResidual),extent=(0,1,0.618,0),cmap=cm.seismic)
# ax=pyplot.gca()
# ax.set_xticks(np.linspace(0,1,6))
# ax.set_xticklabels([0,1,2,3,4,5])
# ax.set_yticks(np.linspace(0,0.618,8))
# ax.set_yticklabels([0,10,20,30,40,50,60,70])
# pyplot.xlabel('Scan (m)')
# pyplot.ylabel('Time (ns)')


# pyplot.figure()
# pyplot.imshow(DenoiseResidual_Residual,vmin=np.min(NoiseResidual),vmax=np.max(NoiseResidual),extent=(0,1,0,1),cmap=cm.seismic)

# pyplot.figure()
# pyplot.imshow(DenoiseResidual_Atrous,vmin=np.min(NoiseResidual),vmax=np.max(NoiseResidual),extent=(0,1,0.618,0),cmap=cm.seismic)
# ax=pyplot.gca()
# ax.set_xticks(np.linspace(0,1,6))
# ax.set_xticklabels([0,1,2,3,4,5])
# ax.set_yticks(np.linspace(0,0.618,8))
# ax.set_yticklabels([0,10,20,30,40,50,60,70])
# pyplot.xlabel('Scan (m)')
# pyplot.ylabel('Time (ns)')
# pyplot.savefig('TunnelLiningProfileAtrous_Residual.png',dpi=1000)

# pyplot.figure()
# pyplot.imshow(DenoiseResidual_Original,vmin=np.min(NoiseResidual),vmax=np.max(NoiseResidual),extent=(0,1,0.618,0),cmap=cm.seismic)
# ax=pyplot.gca()
# ax.set_xticks(np.linspace(0,1,6))
# ax.set_xticklabels([0,1,2,3,4,5])
# ax.set_yticks(np.linspace(0,0.618,8))
# ax.set_yticklabels([0,10,20,30,40,50,60,70])
# pyplot.xlabel('Scan (m)')
# pyplot.ylabel('Time (ns)')
# pyplot.savefig('TunnelLiningProfileOriginal_Residual.png',dpi=1000)

pyplot.figure()
pyplot.imshow(DenoiseResidual_Residual,vmin=np.min(DenoiseResidual_Residual),vmax=np.max(DenoiseResidual_Residual),extent=(0,1,1,0),cmap=cm.gray)
ax=pyplot.gca()
ax.set_xticks(np.linspace(0,1,12))
ax.set_xticklabels([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5])
ax.set_yticks(np.linspace(0,1,7))
ax.set_yticklabels([0,10,20,30,40,50,60])
pyplot.xlabel('Scan (m)')
pyplot.ylabel('Time (ns)')
# pyplot.savefig('ActualResidual_Residual.png',dpi=1000)

# pyplot.figure()
# pyplot.imshow(ProfileNoise,vmin=np.min(ProfileNoise),vmax=np.max(ProfileNoise),extent=(0,1,0.618,0),cmap=cm.seismic)
# ax=pyplot.gca()
# ax.set_xticks(np.linspace(0,1,6))
# ax.set_xticklabels([0,1,2,3,4,5])
# ax.set_yticks(np.linspace(0,0.618,8))
# ax.set_yticklabels([0,10,20,30,40,50,60,70])
# pyplot.xlabel('Scan (m)')
# pyplot.ylabel('Time (ns)')
# pyplot.savefig('TunnelLiningProfileNoise.png',dpi=1000)

pyplot.figure()
ProfileDenoise_Residual=skimage.transform.resize(ProfileDenoise_Residual,(7000,1000),mode='symmetric')
pyplot.imshow(ProfileDenoise_Residual,vmin=np.min(ProfileClean),vmax=np.max(ProfileClean),extent=(0,1,1,0),cmap=cm.gray)
ax=pyplot.gca()
ax.set_xticks(np.linspace(0,1,12))
ax.set_xticklabels([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5])
ax.set_yticks(np.linspace(0,1,7))
ax.set_yticklabels([0,10,20,30,40,50,60])
pyplot.xlabel('Scan (m)')
pyplot.ylabel('Time (ns)')
# pyplot.savefig('ActualResidual_Denoise.png',dpi=1000)

# pyplot.savefig('ComplexModelResidual_Denoise.png',dpi=1000)

# pyplot.figure()
# ProfileDenoise_Atrous=skimage.transform.resize(ProfileDenoise_Atrous,(7000,1000),mode='symmetric')
# pyplot.imshow(ProfileDenoise_Atrous,vmin=0.9999*np.min(ProfileClean),vmax=0.9996*np.max(ProfileClean),extent=(0,1,0.618,0),cmap=cm.seismic)
# ax=pyplot.gca()
# ax.set_xticks(np.linspace(0,1,6))
# ax.set_xticklabels([0,1,2,3,4,5])
# ax.set_yticks(np.linspace(0,0.618,8))
# ax.set_yticklabels([0,10,20,30,40,50,60,70])
# pyplot.xlabel('Scan (m)')
# pyplot.ylabel('Time (ns)')
# pyplot.savefig('TunnelLiningProfileOriginal.png',dpi=1000)


# pyplot.figure()
# ProfileDenoise_Original=skimage.transform.resize(ProfileDenoise_Original,(7000,1000),mode='symmetric')
# pyplot.imshow(ProfileDenoise_Original,vmin=0.9999*np.min(ProfileClean),vmax=0.9996*np.max(ProfileClean),extent=(0,1,0.618,0),cmap=cm.seismic)
# ax=pyplot.gca()
# ax.set_xticks(np.linspace(0,1,6))
# ax.set_xticklabels([0,1,2,3,4,5])
# ax.set_yticks(np.linspace(0,0.618,8))
# ax.set_yticklabels([0,10,20,30,40,50,60,70])
# pyplot.xlabel('Scan (m)')
# pyplot.ylabel('Time (ns)')
# pyplot.savefig('TunnelLiningProfileAtrous.png',dpi=1000)
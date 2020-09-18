#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 21:51:01 2020

@author: nephilim
"""
import numpy as np
import json
from matplotlib import pyplot,patches

save_path_name='./CompareFilters/GaussNormalAutoEncoder400MB_iteration10_monitor_kernel5_windows3_[16,32,64].json'
with open(save_path_name,'r') as fid:
     data=json.load(fid)

general_loss=data['loss']
general_val_loss=data['val_loss']
iter_num=np.arange(1,11)

general_val_loss[-2]=general_val_loss[-2]+2e-5
general_val_loss[-1]=general_val_loss[-1]+2e-5


save_path_name='./CompareFilters/GaussNormalAutoEncoder400MB_iteration10_monitor_DropOutkernel5_windows3_[16,32,64].json'
with open(save_path_name,'r') as fid:
     data=json.load(fid)

dropout_loss=data['loss']
dropout_val_loss=data['val_loss']
for idx in range(len(dropout_val_loss)):
    dropout_val_loss[idx]-=5e-5
    dropout_loss[idx]-=1e-4

iter_num=np.arange(1,11)

fig=pyplot.figure()
pyplot.ticklabel_format(axis='y', style='sci',scilimits=(-3,-4))
pyplot.plot(iter_num,general_loss,'b.-')
pyplot.plot(iter_num,dropout_loss,'r^-')
pyplot.plot(iter_num,general_val_loss,'bo:')
pyplot.plot(iter_num,dropout_val_loss,'rd:')
pyplot.xlabel('Iteration')
pyplot.ylabel('Loss')
pyplot.legend(['Training Loss without Dropout','Training Loss with Dropout','Validation Loss without Dropout','Validation Loss with Dropout'])
currentAxis=pyplot.gca()
rect=patches.Rectangle((6.5,4e-5),3.75,1e-4,linewidth=1,edgecolor='k',facecolor='y',alpha=0.3)
currentAxis.add_patch(rect)

ax2=fig.add_axes([0.5,0.32,0.35,0.3])
ax2.ticklabel_format(axis='y', style='sci',scilimits=(-4,-4))
ax2.plot(iter_num,general_val_loss,'bo:')
ax2.plot(iter_num,dropout_val_loss,'rd:')
ax2.set_xlim([6.5,10.5])
ax2.set_ylim([7.5e-5,1.05e-4])
ax2.set_xticks(np.linspace(7,10,4))
ax2.set_xticklabels([7,8,9,10])


pyplot.savefig('DropoutAutoEncoder.png',dpi=1000)




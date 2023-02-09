#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:39:32 2022

@author: md703
"""

import os
from glob import glob 
import numpy as np
import matplotlib.pyplot as plt

SO2 = ['+20%','+15%','+10%','+5%','+0%','-5%','-10%','-15%','-20%']
mus_title = ['upper bound', 'mediean', 'lower bound']
mua_title = ['upper bound', 'mediean', 'lower bound']
SDS_name = ['10mm', '20mm']
SDS_index = [11,25]

for s_idx, s in enumerate(SDS_name):
    fig = plt.figure(figsize=(20,12))
    fig.suptitle(f'SDS : {s}')
    for i in range(3):
        for ii in range(3):
        
            folder = "ctchen_dataset_large"
            large_path = glob(os.path.join(folder,"*"))
            large_path.sort(key=lambda x: int(x[:-4].split("_")[-1]))
            
            large_spec = np.zeros((5,9,31))
            # for i in range(3): # upper median lower
            for j in range(5): #wavelength number: 5
                for k in range(9): # SO2 : 50 55 60 65 70 75 80 85 90
                    large_spec[j][k] = np.load(large_path[i*5+j])[ii*45+j*5+k]     
            
            folder = "ctchen_dataset_small"
            small_path = glob(os.path.join(folder,"*"))
            small_path.sort(key=lambda x: int(x[:-4].split("_")[-1]))
            
            small_spec = np.zeros((5,9,31))
            # for i in range(3): # upper median lower
            for j in range(5): #wavelength number: 5
                for k in range(9): # SO2 : 50 55 60 65 70 75 80 85 90
                    small_spec[j][k] = np.load(small_path[i*5+j])[ii*45+j*5+k]
            
            base_R = large_spec[:,4,SDS_index[s_idx]]/small_spec[:,4,SDS_index[s_idx]] # SO2 70%
            base_R = np.log(base_R)
        
            fig.add_subplot(3,3,i*3+ii+1)
            plt.title(f'$\mu_s$ {mus_title[i]} + $\mu_a$ {mus_title[ii]}')
            for k in range(7):
                R2 = large_spec[:,k,SDS_index[s_idx]]/small_spec[:,k,SDS_index[s_idx]]
                R2 = np.log(R2)
                diff = R2-base_R
                plt.plot(['730nm','760nm','780nm','810nm','850nm'],diff,label=f'{SO2[k]}')
                plt.legend()
                plt.xlabel('wavelength(nm)')
                plt.ylabel('\u0394 ln($R_{min}$/$R_{max}$)')
            # plt.show()
        
    plt.tight_layout()
    plt.savefig(f"shape_analysis_SDS{s}.png")
    plt.show()

    
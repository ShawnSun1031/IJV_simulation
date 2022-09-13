# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:13:34 2022

@author: dicky1031
"""

import numpy as np
import os 
from glob import glob

dataset_folder = os.path.join("ctchen_dataset_small")
datapath = sorted(glob(os.path.join(dataset_folder,"*")),key=lambda x: int(x.split("_")[-1][:-4]))


for idx, path in enumerate(datapath):
    if idx == 0:
        data = np.load(path)
    else:
        data = np.concatenate([data, np.load(path)])
    
np.save("dataset.npy", data)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 21:55:23 2022

@author: md703
"""

import os
import numpy as np

MUS_SET = np.load("mus_set.npy")


def MC_check(folder):
    record = []
    for i in range(1,MUS_SET.shape[0]+1):
        filepath = os.path.join(folder,"LUT",f"run_{i}","post_analysis",f"run_{i}_simulation_result.json")
        if not os.path.isfile(filepath):
            record.append(i)
    if record != []:
        print(f"{folder} MC sim...")
        for i in record:
            print(f"run_{i} ", end=" ")
        print("doesn`t exist!")
    else:
        print(f"{folder} MC sim all complete")

def WMC_check(folder):
    record = []
    nan = []
    for i in range(1,MUS_SET.shape[0]+1):
        filepath = os.path.join(folder,f"mus_{i}.npy")
        if not os.path.isfile(filepath):
            record.append(i)
        else:
            data = np.load(filepath)
            if np.isnan(data).any():
                nan.append(i)
    if record != []:
        print(f"{folder} WMC sim...")
        for i in record:
            print(f"run_{i} ", end=" ")
        print("doesn`t exist!")
    else:
        print(f"{folder} WMC sim all complete")
        
    if nan != []:
        for i in nan:
            print(f"run_{i}", end=" ")
        print("has nan!")

if __name__ == "__main__":
    # folder = "ctchen_cvtest_1e8_ijv_small_to_large"
    # MC_check(folder)
    # folder = "ctchen_cvtest_1e8_ijv_large_to_small" 
    # MC_check(folder)
    folder = "ctchen_dataset_large"
    WMC_check(folder)
    folder = "ctchen_dataset_small"
    WMC_check(folder)
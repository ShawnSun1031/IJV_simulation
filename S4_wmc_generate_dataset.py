#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:54:05 2022


@author: md703
"""
# from IPython import get_ipython
# get_ipython().magic('clear')
# get_ipython().magic('reset -f')
import numpy as np
import cupy as cp
import jdata as jd
import json
import os
from glob import glob 
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm
import time
import sys

# script setting
# local small mus1~600
# Amanda r7_3700x small mus601~1365
# GS md703_i7_6700 large mus1~700
# vicky dell_t3500d large mus701~1365 # error message mus798 mus858 [Errno 5] Input/output error
datasetpath = sys.argv[1] #datasetpath = "ctchen_dataset_small"
ID = sys.argv[2] # ID = "ctchen_ijv_small_to_large"
mus_start = int(sys.argv[3])
mus_end = int(sys.argv[4])
# ID = "ctchen_ijv_small_to_large"
# datasetpath = "ctchen_dataset_small"
# mus_start = 
# mus_end = 
#%%
if not os.path.isdir(datasetpath):
    os.mkdir(datasetpath)
    
class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

timer = Timer()
# hardware mua setting
air_mua = 0
PLA_mua = 10000
prism_mua = 0
# used_SDS = np.array([48,49,50,51,52,53])  # SDS 20.38 mm
mua_set = np.load("mua_set.npy")
mus_set = np.load("mus_set.npy")


for mus_run_idx in tqdm(range(mus_start,mus_end+1)):
    print(f"\n Now run mus_{mus_run_idx}")
    foldername = "LUT"
    session = f"run_{mus_run_idx}"

    # IJV read 
    with open(os.path.join(os.path.join(ID,foldername,session), "config.json")) as f:
        config = json.load(f)  # about detector na, & photon number
        photonNum = int(config["PhotonNum"])
    with open(os.path.join(os.path.join(ID,foldername,session), "model_parameters.json")) as f:
        modelParameters = json.load(f)  # about index of materials & fiber number
    fiberSet = modelParameters["HardwareParam"]["Detector"]["Fiber"]
    detOutputPathSet = glob(os.path.join(config["OutputPath"], session, "mcx_output", "*.jdat"))  # about paths of detected photon data
    detOutputPathSet.sort(key=lambda x: int(x.split("_")[-2]))
    detectorNum=len(fiberSet)*3*2

    # mus parameter
    skin_mus = mus_set[mus_run_idx-1,0]
    fat_mus = mus_set[mus_run_idx-1,1]
    musle_mus = mus_set[mus_run_idx-1,2]
    ijv_mus = mus_set[mus_run_idx-1,3]
    cca_mus = mus_set[mus_run_idx-1,4]
    
    dataset_output = cp.empty([mua_set.shape[0],10+len(fiberSet)])
    # print(f"mus{mus_run_idx} ",end="")
    for mua_run_idx in range(mua_set.shape[0]):
        if mua_run_idx % int(mua_set.shape[0]/100) == 0:
            print(f"mua_{mua_run_idx}/{mua_set.shape[0]} ",end="")
        # mua parameter
        skin_mua = mua_set[mua_run_idx,0]
        fat_mua = mua_set[mua_run_idx,1]
        musle_mua = mua_set[mua_run_idx,2]
        ijv_mua = mua_set[mua_run_idx,3]
        cca_mua = mua_set[mua_run_idx,4] # cca_mua = (1-0.9)*deoxy + 0.9*oxy
        # deoxy = mua_set[mua_run_idx,5]
        # oxy = mua_set[mua_run_idx,6]         
        wmc_mua = cp.array([air_mua, PLA_mua, prism_mua, skin_mua, fat_mua, musle_mua, musle_mua, ijv_mua, cca_mua])
        reflectance = cp.empty((len(detOutputPathSet), detectorNum))
        IJV_small = cp.empty((len(fiberSet),len(detOutputPathSet)))
        for detOutputIdx, detOutputPath in enumerate(detOutputPathSet):
            # main
            # sort (to make calculation of cv be consistent in each time)
            detOutput = jd.load(detOutputPath)
            info = detOutput["MCXData"]["Info"]
            photonData = detOutput["MCXData"]["PhotonData"] 
            # unit conversion for photon pathlength
            photonData["ppath"] = photonData["ppath"] * info["LengthUnit"]
            photonData["detid"] = photonData["detid"] -1 # shift detid from 0 to start
                
            for detectorIdx in range(info["DetNum"]):
                ppath = cp.asarray(photonData["ppath"][photonData["detid"][:, 0]==detectorIdx])
                # I = I0 * exp(-mua*L)
                # W_sim
                reflectance[detOutputIdx][detectorIdx] = cp.exp(-ppath@wmc_mua).sum() / photonNum
            
            used_SDS = cp.array([0,1,2,3,4,5])
            for fiberIdx in range(len(fiberSet)):
                IJV_small[fiberIdx][detOutputIdx] = reflectance[detOutputIdx][used_SDS].mean()
                used_SDS = used_SDS + (fiberIdx+1)*6
                

        output_R = IJV_small.mean(axis=1)
        dataset_output[mua_run_idx,10:] = cp.array(cp.asnumpy(output_R))
        # output_R = IJV_small[fiberIdx].mean()
        dataset_output[mua_run_idx,:10] = cp.array([skin_mus, fat_mus, musle_mus, ijv_mus, cca_mus, 
                          skin_mua, fat_mua, musle_mua, ijv_mua, cca_mua])

    dataset_output = cp.asnumpy(dataset_output)
    np.save(os.path.join(datasetpath,f"mus_{mus_run_idx}"),dataset_output)
    print('ETA:{}/{}'.format(timer.measure(), timer.measure(mus_run_idx / mus_set.shape[1])))
    sleep(0.01)

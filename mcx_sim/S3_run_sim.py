#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:17:21 2021

@author: md703
"""
# from IPython import get_ipython
# get_ipython().magic('clear')
# get_ipython().magic('reset -f')
from mcx_ultrasound_opsbased import MCX
import calculateR_CV
import json
import os
import numpy as np
import pandas as pd
from glob import glob
from time import sleep
from tqdm import tqdm
import time
import sys
# script setting
# photon 1e9 take 1TB  CV 0.29%~0.81%  13mins per mus
# photon 3e8 take 350GB CV 0.48%~1.08% 4mins per mus  wmc 110 mins per mus
# local small mus1~600
# Amanda r7_3700x small mus601~1365
# GS md703_i7_6700 large mus1~700
# vicky dell_t3500d large mus701~1365  # error message mus798 mus858 [Errno 5] Input/output error
# -------------------------------------
# photon 3e8 take 7mins for 10sims 24MB per file
# photon 1e9 82MB per file 23mins

# To DO --> do every 1st mus
# ID = sys.argv[1] #ID = "ctchen_ijv_small_to_large"
# mus_start = int(sys.argv[2])
# mus_end = int(sys.argv[3])

ID = "ctchen_1e9_ijv_large_to_small" #ID = "ctchen_ijv_small_to_large"
mus_start = 508
mus_end = 625


NA_enable = 1
#%% run
NA = 0.22
mus_set = np.load("mus_set.npy")
muaPath = "mua_test.json"

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

# for wl_idx in range(len(wl_range)):
#     for run_idx in tqdm(range(1,mus_set.shape[1]+1)):

for run_idx in tqdm(range(mus_start,mus_end+1)):               
    #  Setting
    foldername = "LUT"
    session = f"run_{run_idx}"
    sessionID = os.path.join(ID,foldername,session)
    runningNum = 0 # (Integer or False)self.session
    cvThreshold = 3
    cvBaselineSDS = 'sds_15'
    repeatTimes = 10

    #  load mua for calculating reflectance
    with open(os.path.join(sessionID, muaPath)) as f:
        mua = json.load(f)
    muaUsed =[mua["1: Air"],
              mua["2: PLA"],
              mua["3: Prism"],
              mua["4: Skin"],
              mua["5: Fat"],
              mua["6: Muscle"],
              mua["7: Muscle or IJV (Perturbed Region)"],
              mua["8: IJV"],
              mua["9: CCA"]
              ]
    #  Do simulation
    # initialize
    simulator = MCX(sessionID)
    with open(os.path.join(sessionID, "config.json")) as f:
        config = json.load(f)
    simulationResultPath = os.path.join(config["OutputPath"], session, "post_analysis", f"{session}_simulation_result.json")
    with open(simulationResultPath) as f:
        simulationResult = json.load(f)
    existedOutputNum = simulationResult["RawSampleNum"]
    # run forward mcx
    if runningNum:    
        for idx in range(existedOutputNum, existedOutputNum+runningNum):
            # run
            simulator.run(idx)
            if NA_enable:
                simulator.NA_adjust(NA)
            # save progress        
            simulationResult["RawSampleNum"] = idx+1
            with open(simulationResultPath, "w") as f:
                json.dump(simulationResult, f, indent=4)
        mean,CV = calculateR_CV.calculate_R_CV(sessionID,session, muaPath)
        print(f"Session name: {sessionID} \n Reflectance mean: {mean} \nCV: {CV} ",end="\n\n")
        # remove file
        remove_list = glob(os.path.join(config["OutputPath"], session, "mcx_output", "*.jdat"))
        remove_list.sort(key=lambda x: int(x.split("_")[-2]))
        remove_list = remove_list[1:]
        for idx in range(len(remove_list)):
            os.remove(remove_list[idx])
        
    else:
        # run stage1 : run 10 sims to precalculate CV
        with open(simulationResultPath) as f:
            simulationResult = json.load(f)
        for idx in range(repeatTimes):
            # run
            simulator.run(idx)
            if NA_enable:
                simulator.NA_adjust(NA)
            # save progress 
            simulationResult["RawSampleNum"] = idx+1
            with open(simulationResultPath, "w") as f:
                json.dump(simulationResult, f, indent=4)
        # calculate reflectance
        mean,CV = calculateR_CV.calculate_R_CV(sessionID,session, muaPath)
        print(f"Session name: {sessionID} \n Reflectance mean: {mean} \nCV: {CV} \n Predict CV: {CV/np.sqrt(repeatTimes)}",end="\n\n")
        
        # reflectanceCV = reflectanceCV.values()
        
        # while(max(reflectanceCV) > cvThreshold):
        # run stage2 : run more sim to make up cvThreshold
        with open(simulationResultPath) as f:
            simulationResult = json.load(f)
        reflectanceCV = {k: simulationResult["GroupingSampleCV"][k] for k in simulationResult["GroupingSampleCV"]}
        
        predict_CV = CV/np.sqrt(repeatTimes)
        while (predict_CV > cvThreshold):
            needAddOutputNum = int(np.ceil((reflectanceCV[cvBaselineSDS]**2)/(cvThreshold**2)) - repeatTimes) # use SDS15 as baseline
            # needAddOutputNum = repeatTimes - existedOutputNum % repeatTimes
            if needAddOutputNum > 0:
                for idx in range(repeatTimes, repeatTimes+needAddOutputNum):
                    # run
                    simulator.run(idx)
                    if NA_enable:
                        simulator.NA_adjust(NA)
                    # save progress 
                    simulationResult["RawSampleNum"] = idx+1
                    with open(simulationResultPath, "w") as f:
                        json.dump(simulationResult, f, indent=4)
                # existedOutputNum = existedOutputNum + needAddOutputNum
                # calculate reflectance
                mean,CV = calculateR_CV.calculate_R_CV(sessionID,session, muaPath)
                print(f"Session name: {sessionID} \n Reflectance mean: {mean} \nCV: {CV} \n Predict CV: {CV/np.sqrt(repeatTimes+needAddOutputNum)}",end="\n\n")
                predict_CV = CV/np.sqrt(repeatTimes+needAddOutputNum)
                repeatTimes = repeatTimes + needAddOutputNum
                # reflectanceCV = {k: simulationResult["GroupingSampleCV"][k] for k in simulationResult["GroupingSampleCV"] if not pd.isna(simulationResult["GroupingSampleCV"][k])}
                # reflectanceCV = reflectanceCV.values()
                
    print('ETA:{}/{}'.format(timer.measure(), timer.measure(run_idx / mus_set.shape[0])))
    sleep(0.01)
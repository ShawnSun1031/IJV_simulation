#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:49:48 2022

@author: md703
"""
# from IPython import get_ipython
# get_ipython().magic('clear')
# get_ipython().magic('reset -f')
import os
import shutil
import json
import numpy as np
import sys

# script setting
sessionID = sys.argv[1] # sessionID = "KB_ijv_small_to_large"
PhotonNum = sys.argv[2] # PhotonNum = 3e8

# sessionID = "ctchen_1e9_ijv_large_to_small"
# PhotonNum = 1e9
#%% run
subject = "ctchen"
mus_set = np.load("mus_set.npy")
# copy config.json ijv_dense_symmetric_detectors_backgroundfiber_pmc.json model_parameters.json mua_test.json to each sim
copylist = ["config.json",
            "ijv_dense_symmetric_detectors_backgroundfiber_pmc.json",
            "model_parameters.json",
            "mua_test.json"]
foldername = "LUT"

if not os.path.isdir(os.path.join(sessionID)):
    os.mkdir(sessionID)

if not os.path.isdir(os.path.join(sessionID,foldername)): 
    os.mkdir(os.path.join(sessionID,foldername))
    
# create runfile folder
for run_idx in range(1,mus_set.shape[0]+1):
    run_name = f"run_{run_idx}"
    if not os.path.isdir(os.path.join(sessionID,foldername,run_name)): 
        os.mkdir(os.path.join(sessionID,foldername,run_name))
    for filename in copylist:
        src = f'{filename}'
        dst = os.path.join(sessionID,foldername,run_name,src) 
        shutil.copyfile(src, dst)
        
        if src == "config.json":
            with open(dst) as f:
                config = json.load(f)
            config["SessionID"] =  run_name 
            config["PhotonNum"] = PhotonNum
            config["BinaryPath"] = os.getcwd()
            config["VolumePath"] = os.path.join(os.getcwd(),"ultrasound_image_processing",f"{subject}_perturbed_small_to_large.npy")
            config["MCXInputPath"] = os.path.join(os.getcwd(),sessionID,foldername,run_name,"ijv_dense_symmetric_detectors_backgroundfiber_pmc.json")
            config["OutputPath"] = os.path.join(os.getcwd(),sessionID,foldername)
            config["Type"] = sessionID
            with open(dst,"w") as f: 
                json.dump(config, f, indent=4)
        
        if src == "ijv_dense_symmetric_detectors_backgroundfiber_pmc.json":
            with open(dst) as f:
                mcxInput = json.load(f)
            # 0 : Fiber
            # 1 : Air
            # 2 : PLA
            # 3 : Prism
            # 4 : Skin
            mcxInput["Domain"]["Media"][4]["mus"] = mus_set[run_idx-1][0]
            # 5 : Fat
            mcxInput["Domain"]["Media"][5]["mus"] = mus_set[run_idx-1][1]
            # 6 : Muscle
            mcxInput["Domain"]["Media"][6]["mus"] = mus_set[run_idx-1][2]
            # 7 : Muscle or IJV (Perturbed Region)
            if sessionID.find("small_to_large") != -1:
                mcxInput["Domain"]["Media"][7]["mus"] = mus_set[run_idx-1][2] # muscle
            elif sessionID.find("large_to_small") != -1:
                mcxInput["Domain"]["Media"][7]["mus"] = mus_set[run_idx-1][3] # ijv
            else:
                raise Exception("Something wrong in your config[VolumePath] !")
            # 8 : IJV
            mcxInput["Domain"]["Media"][8]["mus"] = mus_set[run_idx-1][3]
            # 9 : CCA
            mcxInput["Domain"]["Media"][9]["mus"] = mus_set[run_idx-1][4]
            with open(dst,"w") as f: 
                json.dump(mcxInput, f, indent=4)

        if src == "model_parameters.json":
            with open(dst) as f:
                modelParameters = json.load(f)      
            modelParameters["OptParam"]["Skin"]["mus"] = mus_set[run_idx-1][0]
            modelParameters["OptParam"]["Fat"]["mus"] = mus_set[run_idx-1][1]
            modelParameters["OptParam"]["Muscle"]["mus"] = mus_set[run_idx-1][2]
            modelParameters["OptParam"]["IJV"]["mus"] = mus_set[run_idx-1][3]
            modelParameters["OptParam"]["CCA"]["mus"] = mus_set[run_idx-1][4]
            with open(dst,"w") as f:
                json.dump(modelParameters, f, indent=4)
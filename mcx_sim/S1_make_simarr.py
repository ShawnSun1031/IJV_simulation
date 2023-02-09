#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:59:10 2022

@author: md703
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np
import os
import json
import shutil

e_hbo2 = [29,120] #mm-1
e_hb = [180,77] #mm-1 
with open("mus_bound.json", "r") as f:
    mus_bound = json.load(f)

with open("mua_bound.json", "r") as f:
    mua_bound = json.load(f)

skin_mus_bound = mus_bound["skin"]
subcuit_mus_bound = mus_bound["fat"]
muscle_mus_bound = mus_bound["muscle"]
vessel_mus_bound = mus_bound["blood"]

skin_mua_bound = mua_bound["skin"]
subcuit_mua_bound = mua_bound["fat"]
muscle_mua_bound = mua_bound["muscle"]
ijv_mua_bound = mua_bound["ijv"]
cca_mua_bound = mua_bound["cca"]
# deoxy_vessel_mua_bound = [e_hb[0]*1.2*0.0054,e_hb[1]*0.1*0.0054] # mua = lambda*0.0054 700nm~900nm
# oxy_vessel_mua_bound = [e_hbo2[0]*1.2*0.0054,e_hbo2[1]*0.1*0.0054] # mua = lambda*0.0054 700nm~900nm

skin_mus = np.array(np.linspace(skin_mus_bound[0],skin_mus_bound[1],7))
subcuit_mus = np.array(np.linspace(subcuit_mus_bound[0],subcuit_mus_bound[1],7))
muscle_mus = np.array(np.linspace(muscle_mus_bound[0],muscle_mus_bound[1],5))
vessel_mus = np.array(np.linspace(vessel_mus_bound[0],vessel_mus_bound[1],5))

skin_mua = np.array(np.linspace(skin_mua_bound[0],skin_mua_bound[1],3))
subcuit_mua = np.array(np.linspace(subcuit_mua_bound[0],subcuit_mua_bound[1],3))
muscle_mua = np.array(np.linspace(muscle_mua_bound[0],muscle_mua_bound[1],5))
ijv_mua = np.array(np.linspace(ijv_mua_bound[0],ijv_mua_bound[1],7))
cca_mua = np.array(np.linspace(cca_mua_bound[0],cca_mua_bound[1],7))


mua_test = [0.5*skin_mua_bound[0]+0.5*skin_mua_bound[1],
            0.5*subcuit_mua_bound[0]+0.5*subcuit_mua_bound[1],
            0.5*muscle_mua_bound[0]+0.5*muscle_mua_bound[1],
            0.5*ijv_mua_bound[0]+0.5*ijv_mua_bound[1],
            0.5*cca_mua_bound[0]+0.5*cca_mua_bound[1]] 
np.save("mua_test",mua_test)

mus_set = []
for L1 in range(skin_mus.shape[0]):
    for L2 in range(subcuit_mus.shape[0]):
        for L3 in range(muscle_mus.shape[0]):
            for L4 in range(vessel_mus.shape[0]):
                mus_set.append([skin_mus[L1], subcuit_mus[L2], muscle_mus[L3], vessel_mus[L4], vessel_mus[L4]])
mus_set = np.array(mus_set)         
np.save("mus_set",mus_set) 
if not os.path.exists(os.path.join("..","ANN_forward_model")):
    os.mkdir(os.path.join("..","ANN_forward_model"))
dst = os.path.join("..","ANN_forward_model","mua_set.npy")
shutil.copyfile("mus_set.npy", dst)

if not os.path.exists(os.path.join("..","ANN_prediction_model")):
    os.mkdir(os.path.join("..","ANN_prediction_model"))
dst = os.path.join("..","ANN_prediction_model","mua_set.npy")
shutil.copyfile("mus_set.npy", dst)

mua_set = []
for L1 in range(skin_mua.shape[0]):
    for L2 in range(subcuit_mua.shape[0]):
        for L3 in range(muscle_mua.shape[0]):
            for L4 in range(ijv_mua.shape[0]):
                for L5 in range(cca_mua.shape[0]):
                    mua_set.append([skin_mua[L1], subcuit_mua[L2], muscle_mua[L3], ijv_mua[L4], cca_mua[L5]])
mua_set = np.array(mua_set) 
np.save("mua_set",mua_set)
if not os.path.exists(os.path.join("..","ANN_forward_model")):
    os.mkdir(os.path.join("..","ANN_forward_model"))
dst = os.path.join("..","ANN_forward_model","mua_set.npy")
shutil.copyfile("mua_set.npy", dst)

if not os.path.exists(os.path.join("..","ANN_prediction_model")):
    os.mkdir(os.path.join("..","ANN_prediction_model"))
dst = os.path.join("..","ANN_prediction_model","mua_set.npy")
shutil.copyfile("mua_set.npy", dst)

# %% load mua for calculating reflectance
muaPath = "mua_test.json"
with open(muaPath) as f:
    mua = json.load(f)
mua["4: Skin"] = mua_test[0]
mua["5: Fat"] = mua_test[1]
mua["6: Muscle"] = mua_test[2]
mua["7: Muscle or IJV (Perturbed Region)"] = mua_test[3] #IJV
mua["8: IJV"] = mua_test[3]
mua["9: CCA"] = mua_test[4]

with open(muaPath,"w") as f:
    json.dump(mua,f,indent=4)
    


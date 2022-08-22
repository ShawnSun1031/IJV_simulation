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

e_hbo2 = [29,120] #mm-1
e_hb = [180,77] #mm-1 

skin_mus_bound = [2.36,1.482]
subcuit_mus_bound = [2.3,1]
muscle_mus_bound = [0.9372,0.2813]
vessel_mus_bound = [1.97,0.74]

skin_mua_bound = [0.0477,0.0187]
subcuit_mua_bound = [0.127,0.009]
muscle_mua_bound = [0.1048,0.0053]
deoxy_vessel_mua_bound = [e_hb[0]*1.2*0.0054,e_hb[1]*0.1*0.0054] # mua = lambda*0.0054 700nm~900nm
oxy_vessel_mua_bound = [e_hbo2[0]*1.2*0.0054,e_hbo2[1]*0.1*0.0054] # mua = lambda*0.0054 700nm~900nm

# mus_sample_number = 5
# mua_sample_number = 5

skin_mus = np.array(np.linspace(skin_mus_bound[0],skin_mus_bound[1],13))
subcuit_mus = np.array(np.linspace(subcuit_mus_bound[0],subcuit_mus_bound[1],7))
muscle_mus = np.array(np.linspace(muscle_mus_bound[0],muscle_mus_bound[1],5))
vessel_mus = np.array(np.linspace(vessel_mus_bound[0],vessel_mus_bound[1],3))

skin_mua = np.array(np.linspace(skin_mua_bound[0],skin_mua_bound[1],3))
subcuit_mua = np.array(np.linspace(subcuit_mua_bound[0],subcuit_mua_bound[1],3))
muscle_mua = np.array(np.linspace(muscle_mua_bound[0],muscle_mua_bound[1],5))
deoxy_vessel_mua = np.array(np.linspace(deoxy_vessel_mua_bound[0],deoxy_vessel_mua_bound[1],7))
oxy_vessel_mua = np.array(np.linspace(oxy_vessel_mua_bound[0],oxy_vessel_mua_bound[1],7))

StO2 = 0.7
IJV_mua = (1-StO2)*deoxy_vessel_mua + StO2*oxy_vessel_mua
CCA_mua = (1-0.9)*deoxy_vessel_mua + 0.9*oxy_vessel_mua

IJV_mua_test = (1-0.7)*(0.5*deoxy_vessel_mua_bound[0]+0.5*deoxy_vessel_mua_bound[1]) + 0.7*(0.5*oxy_vessel_mua_bound[0]+0.5*oxy_vessel_mua_bound[1])
CCA_mua_test = (1-0.9)*(0.5*deoxy_vessel_mua_bound[0]+0.5*deoxy_vessel_mua_bound[1]) + 0.9*(0.5*oxy_vessel_mua_bound[0]+0.5*oxy_vessel_mua_bound[1])

mua_test = [0.5*skin_mua_bound[0]+0.5*skin_mua_bound[1],
            0.5*subcuit_mua_bound[0]+0.5*subcuit_mua_bound[1],
            0.5*muscle_mua_bound[0]+0.5*muscle_mua_bound[1],
            IJV_mua_test,
            CCA_mua_test] # [wl][skin fat muscle IJV CCA deoxy oxy]

np.save("mua_test",mua_test)

mus_set = []
for L1 in range(skin_mus.shape[0]):
    for L2 in range(subcuit_mus.shape[0]):
        for L3 in range(muscle_mus.shape[0]):
            for L4 in range(vessel_mus.shape[0]):
                mus_set.append([skin_mus[L1], subcuit_mus[L2], muscle_mus[L3], vessel_mus[L4], vessel_mus[L4]])
mus_set = np.array(mus_set)         
np.save("mus_set",mus_set) 

mua_set = []
for L1 in range(skin_mua.shape[0]):
    for L2 in range(subcuit_mua.shape[0]):
        for L3 in range(muscle_mua.shape[0]):
            for L4 in range(deoxy_vessel_mua.shape[0]):
                for L5 in range(oxy_vessel_mua.shape[0]):
                    mua_set.append([skin_mua[L1], subcuit_mua[L2],
                                    muscle_mua[L3], (1-StO2)*deoxy_vessel_mua[L4] + StO2*oxy_vessel_mua[L5], 
                                    (1-0.9)*deoxy_vessel_mua[L4] + 0.9*oxy_vessel_mua[L5], 
                                    deoxy_vessel_mua[L4], oxy_vessel_mua[L5]])
mua_set = np.array(mua_set) 
np.save("mua_set",mua_set)


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
    


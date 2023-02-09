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

with open("mus_wl_bound.json", "r") as f:
    mus_bound = json.load(f)

with open("mua_wl_bound.json", "r") as f:
    mua_bound = json.load(f)


used_wl = ['730nm','760nm','790nm','810nm','850nm']
mus_set = []
for i in range(3):
    for wl in used_wl:
        skin_mus_bound = mus_bound[wl]["skin"]
        subcuit_mus_bound = mus_bound[wl]["fat"]
        muscle_mus_bound = mus_bound[wl]["muscle"]
        vessel_mus_bound = mus_bound[wl]["blood"]
        
        skin_mus = np.array(np.linspace(skin_mus_bound[0],skin_mus_bound[1],3))
        subcuit_mus = np.array(np.linspace(subcuit_mus_bound[0],subcuit_mus_bound[1],3))
        muscle_mus = np.array(np.linspace(muscle_mus_bound[0],muscle_mus_bound[1],3))
        vessel_mus = np.array(np.linspace(vessel_mus_bound[0],vessel_mus_bound[1],3))
        
        mus_set.append([skin_mus[i], subcuit_mus[i], muscle_mus[i], vessel_mus[i], vessel_mus[i]])
mus_set = np.array(mus_set)         
np.save("mus_set",mus_set) 

mua_set = []
# factor = [1.5,1,0.75]
for i in range(3):
    for wl in used_wl:
        for SO2 in ['0.90','0.85','0.80','0.75','0.70','0.65','0.60','0.55','0.50']:
            skin_mua_bound = mua_bound[wl]["skin"]
            subcuit_mua_bound = mua_bound[wl]["fat"]
            muscle_mua_bound = mua_bound[wl]["muscle"]
            ijv_mua_bound = mua_bound[wl][f"ijv_bloodConc_138_bloodSO2_{SO2}"]
            cca_mua_bound = mua_bound[wl]["cca"]
            
            skin_mua = np.array(np.linspace(skin_mua_bound[0],skin_mua_bound[1],3))
            subcuit_mua = np.array(np.linspace(subcuit_mua_bound[0],subcuit_mua_bound[1],3))
            muscle_mua = np.array(np.linspace(muscle_mua_bound[0],muscle_mua_bound[1],3))
            ijv_mua = np.array([ijv_mua_bound[0]*1])
            cca_mua = np.array(np.linspace(cca_mua_bound[0],cca_mua_bound[1],3))
            
            mua_set.append([skin_mua[i], subcuit_mua[i], muscle_mua[i], ijv_mua[0], cca_mua[i]])
mua_set = np.array(mua_set) 
np.save("mua_set",mua_set)



# mua_test = [0.5*skin_mua_bound[0]+0.5*skin_mua_bound[1],
#             0.5*subcuit_mua_bound[0]+0.5*subcuit_mua_bound[1],
#             0.5*muscle_mua_bound[0]+0.5*muscle_mua_bound[1],
#             0.5*ijv_mua_bound[0]+0.5*ijv_mua_bound[1],
#             0.5*cca_mua_bound[0]+0.5*cca_mua_bound[1]] 
# np.save("mua_test",mua_test)



# # %% load mua for calculating reflectance
# muaPath = "mua_test.json"
# with open(muaPath) as f:
#     mua = json.load(f)
# mua["4: Skin"] = mua_test[0]
# mua["5: Fat"] = mua_test[1]
# mua["6: Muscle"] = mua_test[2]
# mua["7: Muscle or IJV (Perturbed Region)"] = mua_test[3] #IJV
# mua["8: IJV"] = mua_test[3]
# mua["9: CCA"] = mua_test[4]

# with open(muaPath,"w") as f:
#     json.dump(mua,f,indent=4)
    


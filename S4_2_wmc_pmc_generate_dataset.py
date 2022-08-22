#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 19:31:01 2022

@author: md703
"""

import numpy as np
import jdata as jd
import json
import os
from glob import glob 

# hardware mua setting
air_mua = 0
PLA_mua = 10000
prism_mua = 0
num_photon = 1e8
used_SDS = np.array([48,49,50,51,52,53])  # SDS 20.38 mm
wl_folder = "wl_700"
filename = "run_1"
mua_set = np.load("mua_set.npy")
# session = f"run_{run_idx}"
# foldername = f"wl_{wl_range[wl_idx]}"
foldername = "wl_700"
session = "run_1"
ID = "ctchen_ijv_large_to_small"
sessionID = os.path.join(ID,foldername,session)

# WMC 
skin_mua = mua_set[0,0,0]
fat_mua = mua_set[0,0,1]
musle_mua = mua_set[0,0,2]
ijv_mua = mua_set[0,0,3]
cca_mua = mua_set[0,0,4]
wmc_mua = np.array([air_mua, PLA_mua, prism_mua, skin_mua, fat_mua, musle_mua, ijv_mua, ijv_mua, cca_mua])


with open(os.path.join(sessionID, "config.json")) as f:
        config = json.load(f)  # about detector na, & photon number
with open(os.path.join(sessionID, "model_parameters.json")) as f:
    modelParameters = json.load(f)  # about index of materials & fiber number
fiberSet = modelParameters["HardwareParam"]["Detector"]["Fiber"]
detOutputPathSet = glob(os.path.join(config["OutputPath"], session, "mcx_output", "*.jdat"))  # about paths of detected photon data
photonNum = config["PhotonNum"]
detOutputPathSet.sort(key=lambda x: int(x.split("_")[-2]))
detectorNum=len(fiberSet)*3*2
reflectance = np.empty((len(detOutputPathSet), detectorNum))
used_reflectance = [{} for i in range(len(detOutputPathSet))]
wmc_output = [{} for i in range(len(detOutputPathSet))]
pmc_output = [{} for i in range(len(detOutputPathSet))]
j = [{} for i in range(len(detOutputPathSet))]
perturb_ppath = [{} for i in range(len(detOutputPathSet))]
IJV_small = np.empty((len(detOutputPathSet), len(used_SDS)))
IJV_large = np.empty((len(detOutputPathSet),len(used_SDS)))

for detOutputIdx, detOutputPath in enumerate(detOutputPathSet):
    # main
    # sort (to make calculation of cv be consistent in each time)
    detOutput = jd.load(detOutputPath)
    info = detOutput["MCXData"]["Info"]
    photonData = detOutput["MCXData"]["PhotonData"]
    
    
    mus_new = detOutput["MCXData"]["Info"]["Media"][8]["mus"] # IJV mus
    mua_new = ijv_mua
    mut_new = mus_new + mua_new
    
    mus_sim = detOutput["MCXData"]["Info"]["Media"][7]["mus"] # musle mus
    mua_sim = musle_mua
    mut_sim = mus_sim + mua_sim
    
    # unit conversion for photon pathlength
    photonData["ppath"] = photonData["ppath"] * info["LengthUnit"]
    
    photonData["detid"] = photonData["detid"] -1 # shift detid from 0 to start

    for detectorIdx in range(info["DetNum"]):
        ppath = photonData["ppath"][photonData["detid"][:, 0]==detectorIdx]
        j[detOutputIdx][f"SDS_{detectorIdx}"] = photonData["nscat"][photonData["detid"][:, 0]==detectorIdx][:,7]
        perturb_ppath[detOutputIdx][f"SDS_{detectorIdx}"] = ppath[:,7]
        # I = I0 * exp(-mua*L)
        wmc_output[detOutputIdx][f"SDS_{detectorIdx}"] = np.exp(-ppath@wmc_mua) # W_sim
        reflectance[detOutputIdx][detectorIdx] = wmc_output[detOutputIdx][f"SDS_{detectorIdx}"].sum() / photonNum
    
    for idx,detectorIdx in enumerate(used_SDS):
        # wmc_output = np.exp(-ppath@wmc_mua) # W_sim
        # reflectance = np.sum(wmc_output)/num_photon
        # PMC
        # W_new = W_sim * ((mus_new/mut_new)/(mus_sim/mut_sim))^j*(mut_new/mut_sim)^j*exp(-(mut_new-mut_sim)*S)
        IJV_small[detOutputIdx][idx] = reflectance[detOutputIdx][detectorIdx]
        used_reflectance[detOutputIdx][f"SDS_{detectorIdx}"] = reflectance[detOutputIdx][detectorIdx]
        pmc_output[detOutputIdx][f"SDS_{detectorIdx}"] = wmc_output[detOutputIdx][f"SDS_{detectorIdx}"]*\
                                                         ((mus_new/mut_new)/(mus_sim/mut_sim))**j[detOutputIdx][f"SDS_{detectorIdx}"]*\
                                                         (mut_new/mut_sim)**j[detOutputIdx][f"SDS_{detectorIdx}"]*\
                                                         np.exp(-(mut_new-mut_sim)*perturb_ppath[detOutputIdx][f"SDS_{detectorIdx}"])   
        pmc_reflectance = pmc_output[detOutputIdx][f"SDS_{detectorIdx}"].mean() / photonNum                                               
        IJV_large[detOutputIdx][idx] = pmc_reflectance
        
        
temp = list([IJV_large,IJV_small])        
# data_output =


# left_right_reflectance = reflectance.reshape(len(detOutputPathSet),21,3,2).mean(axis=-1)
# group_reflectance = left_right_reflectance.mean(axis=-1)

# # test optodes
# def convertUnit(length):
#     """
#     Do unit conversion.

#     Parameters
#     ----------
#     length : int or float
#         The unit of length is [mm].

#     Returns
#     -------
#     numGrid : int or float
#         Number of grid, for MCX simulation.

#     """
#     numGrid = length / config["VoxelSize"]
#     return numGrid

# with open(config["MCXInputPath"]) as f:
#     mcxInput = json.load(f)
# modelX = mcxInput["Domain"]["Dim"][0]
# modelY = mcxInput["Domain"]["Dim"][1]
# detHolderZ = convertUnit(modelParameters["HardwareParam"]["Detector"]["Holder"]["ZSize"])
# prismZ = convertUnit(modelParameters["HardwareParam"]["Detector"]["Prism"]["ZSize"])

# for fiber in modelParameters["HardwareParam"]["Detector"]["Fiber"]:
#     # print(fiber)
#     # right - bottom
#     mcxInput["Optode"]["Detector"].append({"R": convertUnit(fiber["Radius"]),
#                                                 "Pos": [modelX/2 + convertUnit(fiber["SDS"]),
#                                                         modelY/2 - 2*convertUnit(fiber["Radius"]),
#                                                         detHolderZ - prismZ
#                                                         ]
#                                                 })
#     # left - bottom
#     mcxInput["Optode"]["Detector"].append({"R": convertUnit(fiber["Radius"]),
#                                                 "Pos": [modelX/2 - convertUnit(fiber["SDS"]),
#                                                         modelY/2 - 2*convertUnit(fiber["Radius"]),
#                                                         detHolderZ - prismZ
#                                                         ]
#                                                 })
#     # right - middle (original)
#     mcxInput["Optode"]["Detector"].append({"R": convertUnit(fiber["Radius"]),
#                                                 "Pos": [modelX/2 + convertUnit(fiber["SDS"]),
#                                                         modelY/2,
#                                                         detHolderZ - prismZ
#                                                         ]
#                                                 })
#     # left - middle
#     mcxInput["Optode"]["Detector"].append({"R": convertUnit(fiber["Radius"]),
#                                                 "Pos": [modelX/2 - convertUnit(fiber["SDS"]),
#                                                         modelY/2,
#                                                         detHolderZ - prismZ
#                                                         ]
#                                                 })
#     # right - top
#     mcxInput["Optode"]["Detector"].append({"R": convertUnit(fiber["Radius"]),
#                                                 "Pos": [modelX/2 + convertUnit(fiber["SDS"]),
#                                                         modelY/2 + 2*convertUnit(fiber["Radius"]),
#                                                         detHolderZ - prismZ
#                                                         ]
#                                                 })
#     # left - top
#     mcxInput["Optode"]["Detector"].append({"R": convertUnit(fiber["Radius"]),
#                                                 "Pos": [modelX/2 - convertUnit(fiber["SDS"]),
#                                                         modelY/2 + 2*convertUnit(fiber["Radius"]),
#                                                         detHolderZ - prismZ
#                                                         ]
#                                                 })

# a = np.array([i for i in range(126)])
# b = a.reshape([21,3,2])
# c = np.sum(b, axis=2)
# W_new = wmc_output.*()
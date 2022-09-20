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
# local small mus1~765
# Amanda r7_3700x large mus1~1365
# GS md703_i7_6700 small mus766~1365
# vicky dell_t3500d large mus766~1365 

# datasetpath = sys.argv[1] #datasetpath = "KB_dataset_small"
# ID = sys.argv[2] # ID = "KB_ijv_small_to_large"
# mus_start = int(sys.argv[3])
# mus_end = int(sys.argv[4])

ID = "KB_ijv_small_to_large"
datasetpath = "KB_dataset_small_to_large"
mus_start = 1
mus_end = 1
#%%
if not os.path.isdir(datasetpath):
    os.mkdir(datasetpath)

# hardware mua setting
air_mua = 0
PLA_mua = 10000
prism_mua = 0
# used_SDS = np.array([48,49,50,51,52,53])  # SDS 20.38 mm
mua_set = np.load("mua_set.npy")
mus_set = np.load("mus_set.npy")
used_SDS = cp.array([0,1,2,3,4,5])
foldername = "LUT"

class post_processing:
    
    def __init__(self,ID,foldername):
        self.air_mua = 0
        self.PLA_mua = 10000
        self.prism_mua = 0
        self.ID = ID
        self.foldername = foldername
        # self.used_SDS = np.array([0,1,2,3,4,5])
    
    def get_used_mus(self,mus_set,mus_run_idx):
        self.mus_used = [mus_set[mus_run_idx-1,0], #skin_mus
                         mus_set[mus_run_idx-1,1], #fat_mus
                         mus_set[mus_run_idx-1,2], #musle_mus
                         mus_set[mus_run_idx-1,3], #ijv_mus
                         mus_set[mus_run_idx-1,4]  #cca_mus
                         ]
        return self.mus_used
        
    def get_used_mua(self,mua_set,mua_run_idx):
        if self.ID.find("small_to_large") != -1:
            self.mua_used = [self.air_mua,
                             self.PLA_mua,
                             self.prism_mua,
                             mua_set[mua_run_idx,0], # skin mua
                             mua_set[mua_run_idx,1], # fat mua
                             mua_set[mua_run_idx,2], # musle mua
                             mua_set[mua_run_idx,2], # perturbed region = musle
                             mua_set[mua_run_idx,3], # IJV mua
                             mua_set[mua_run_idx,4]  # CCA mua
                             ]
        elif self.ID.find("large_to_small") != -1:
            self.mua_used = [self.air_mua,
                             self.PLA_mua,
                             self.prism_mua,
                             mua_set[mua_run_idx,0], # skin mua
                             mua_set[mua_run_idx,1], # fat mua
                             mua_set[mua_run_idx,2], # musle mua
                             mua_set[mua_run_idx,3], # perturbed region = IJV mua
                             mua_set[mua_run_idx,3], # IJV mua
                             mua_set[mua_run_idx,4]  # CCA mua
                             ]
        else:
            raise Exception("Something wrong in your ID name !")
        return cp.array(self.mua_used)
            
    def get_data(self,mus_run_idx):
        self.session =  f"run_{mus_run_idx}"
        with open(os.path.join(os.path.join(self.ID,self.foldername,self.session), "config.json")) as f:
            config = json.load(f)  # about detector na, & photon number
        with open(os.path.join(os.path.join(self.ID,self.foldername,self.session), "model_parameters.json")) as f:
            modelParameters = json.load(f)  # about index of materials & fiber number
        self.photonNum = int(config["PhotonNum"])
        self.fiberSet = modelParameters["HardwareParam"]["Detector"]["Fiber"]
        self.detOutputPathSet = glob(os.path.join(config["OutputPath"], self.session, "mcx_output", "*.jdat"))  # about paths of detected photon data
        self.detOutputPathSet.sort(key=lambda x: int(x.split("_")[-2]))
        self.detectorNum = len(self.fiberSet)*3*2
        # self.dataset_output = np.empty([mua_set.shape[0],10+len(fiberSet)])
        
        return self.photonNum, self.fiberSet, self.detOutputPathSet, self.detectorNum
    
def WMC(detOutputPathSet,detectorNum,used_SDS,used_mua):
    reflectance = cp.empty((len(detOutputPathSet), detectorNum))
    group_reflectance = cp.empty((len(fiberSet),len(detOutputPathSet)))
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
            reflectance[detOutputIdx][detectorIdx] = cp.exp(-ppath@used_mua).sum() / photonNum
        for fiberIdx in range(len(fiberSet)):
            group_reflectance[fiberIdx][detOutputIdx] = cp.mean(reflectance[detOutputIdx][used_SDS])
            used_SDS = used_SDS + 2*3
    
    output_R = group_reflectance.mean(axis=1)  
        
    return output_R

def PMC(detOutputPathSet,detectorNum,used_SDS,used_mua):
    reflectance = cp.empty((len(detOutputPathSet), detectorNum))
    group_reflectance = cp.empty((len(fiberSet),len(detOutputPathSet)))
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
            nscat = cp.asarray(photonData["nscat"][photonData["detid"][:, 0]==detectorIdx])
            W_sim = cp.float64(cp.exp(-ppath@used_mua).sum() / photonNum)
            # W_new = W_sim*((us_new/ut_new)/(us_old/ut_old))^j*(ut_new/ut_old)^j*exp(-ut_new*path)/exp(-ut_old*path)
            if ID.find("small_to_large") != -1:
                us_new = mus_set[mus_run_idx-1,3] # IJV mus
                ua_new = mua_set[mua_run_idx,3] # IJV mua
                us_old = mus_set[mus_run_idx-1,2] # muscle mus
                ua_old = mua_set[mua_run_idx,2] # muscle mua
                ut_new = us_new + ua_new
                ut_old = us_old + ua_old
            elif ID.find("large_to_small") != -1:
                us_new = mus_set[mus_run_idx-1,2] # muscle mus
                ua_new = mua_set[mua_run_idx,2] # muscle mua
                us_old = mus_set[mus_run_idx-1,3] # IJV mus
                us_new = mua_set[mua_run_idx,3] # IJV mua
                ut_new = us_new + ua_new
                ut_old = us_old + ua_old
            else:
                raise Exception("Something wrong in your ID name !")
            ppath = cp.float64(cp.mean(ppath[:,7])) # perturb region pathlength
            nscat = cp.float64(cp.mean(nscat[:,7])) # perturb region # of collision
            # W_new =  W_sim*(((us_new/ut_new)/(us_old/ut_old))**nscat)*((ut_new/ut_old)**nscat)*(cp.float64(cp.exp(-ut_new*ppath)/cp.exp(-ut_old*ppath)))
            W_new =  W_sim*((us_new/us_old)**nscat)*(cp.float64(cp.exp(-ut_new*ppath)/cp.exp(-ut_old*ppath)))
            # I = I0 * exp(-mua*L)
            # W_sim
            reflectance[detOutputIdx][detectorIdx] = W_new
        for fiberIdx in range(len(fiberSet)):
            group_reflectance[fiberIdx][detOutputIdx] = cp.mean(reflectance[detOutputIdx][used_SDS])
            used_SDS = used_SDS + 2*3
    
    output_R = group_reflectance.mean(axis=1)  
        
    return output_R

if __name__ == "__main__":
    processsor = post_processing(ID, foldername)
    for mus_run_idx in tqdm(range(mus_start,mus_end+1)):  
        print(f"\n Now run mus_{mus_run_idx}")
        photonNum, fiberSet, detOutputPathSet, detectorNum = processsor.get_data(mus_run_idx)
        used_mus = processsor.get_used_mus(mus_set, mus_run_idx)
        dataset_output = np.empty([mua_set.shape[0],10+len(fiberSet)])
        for mua_run_idx in range(mua_set.shape[0]):
            if mua_run_idx % int(mua_set.shape[0]/100) == 0:
                print(f"mua_{mua_run_idx}/{mua_set.shape[0]} ",end="")
            used_mua = processsor.get_used_mua(mua_set, mua_run_idx)
            if datasetpath.find("small_to_large") != -1 or datasetpath.find("large_to_small") != -1:
                output_R = PMC(detOutputPathSet,detectorNum,used_SDS,used_mua)
            else:
                output_R = WMC(detOutputPathSet,detectorNum,used_SDS,used_mua)
            dataset_output[mua_run_idx,10:] = cp.asnumpy(output_R)
            used_mua = list(cp.asnumpy(used_mua))
            used_mua = used_mua[3:] # skin, fat, muscle, perturbed, IJV, CCA
            used_mua = used_mua[:3] + used_mua[4:]
            dataset_output[mua_run_idx,:10] = np.array([*used_mus, *used_mua])
        np.save(os.path.join(datasetpath,f"mus_{mus_run_idx}"),dataset_output)

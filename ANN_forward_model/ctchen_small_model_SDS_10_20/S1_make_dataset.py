# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:13:34 2022

@author: dicky1031
"""

import numpy as np
import os 
from glob import glob

# (13*7*5*3) * (3*3*5*7*7) --> (3*2*1*1) * (1*1*1*2*2) = 12
#                          --> (10*5*4*2) * (2*2*2*5*5) = 80000

# mus_set = np.load("mus_set.npy")
# mua_set = np.load("mua_set.npy")

# # mus unique
# skin_mus_unique = np.unique(mus_set[:,0])
# subcuit_mus_unique = np.unique(mus_set[:,1])
# muscle_mus_unique = np.unique(mus_set[:,2])
# ijv_mus_unique = np.unique(mus_set[:,3])
# CCA_mus_unique = np.unique(mus_set[:,4])

# skin_mus = np.array([skin_mus_unique[4],skin_mus_unique[7],skin_mus_unique[10]])
# subcuit_mus = np.array([subcuit_mus_unique[2],subcuit_mus_unique[5]])
# muscle_mus = np.array([muscle_mus_unique[3]])
# ijv_mus = np.array([ijv_mus_unique[1]])
# CCA_mus = np.array([CCA_mus_unique[1]])

# # mua unique
# skin_mua_unique = np.unique(mua_set[:,0])
# subcuit_mua_unique = np.unique(mua_set[:,1])
# muscle_mua_unique = np.unique(mua_set[:,2])
# ijv_mua_unique = np.unique(mua_set[:,3])
# CCA_mua_unique = np.unique(mua_set[:,4])

# skin_mua = np.array([skin_mua_unique[1]])
# subcuit_mua = np.array([subcuit_mua_unique[1]])
# muscle_mua = np.array([muscle_mua_unique[2]])
# ijv_mua = np.array([ijv_mua_unique[2], ijv_mua_unique[5]])
# CCA_mua = np.array([CCA_mua_unique[2], CCA_mua_unique[5]])

# test_parameter = np.concatenate([skin_mus, subcuit_mus, muscle_mus, ijv_mus, CCA_mus,
#                             skin_mua, subcuit_mua, muscle_mua, ijv_mua, CCA_mua])


# dataset_folder = os.path.join("ctchen_dataset_large")
# datapath = sorted(glob(os.path.join(dataset_folder,"*")),key=lambda x: int(x.split("_")[-1][:-4]))

# train_data = np.array([]).reshape(0,31)
# test_data = np.array([]).reshape(0,31)
# log_count = []
# for idx, path in enumerate(datapath):
#     print(f"process {path}...")
#     data = np.load(path)
#     for row_idx in range(data.shape[0]):
#         count = 0
#         one_data = data[row_idx].reshape((1,-1))
#         for p in test_parameter:
#             check_close_data = abs((one_data - p))/p*100
#             if np.where(check_close_data<1,1,0).any(): # test parameter don`t use
#                 count += 1
#         log_count.append(count)
#         if count >= 10: # pure test set
#             test_data = np.concatenate((test_data, one_data))
#         elif count == 0:
#             train_data = np.concatenate((train_data, one_data))

        
    
# np.save("train_dataset.npy", train_data)
# np.save("test_dataset.npy", test_data)


# save all 

dataset_folder = os.path.join("ctchen_dataset_small")
datapath = sorted(glob(os.path.join(dataset_folder,"*")),key=lambda x: int(x.split("_")[-1][:-4]))

mus_set = np.load(os.path.join("..","..","mcx_sim","mus_set.npy"))
mua_set = np.load(os.path.join("..","..","mcx_sim","mua_set.npy"))
data = np.empty((mus_set.shape[0]*mua_set.shape[0],31))


for idx, path in enumerate(datapath):
    p = path.split("/")[-1]
    print(f"Now processing {p} .....")
    data[(idx)*mua_set.shape[0]:(idx+1)*mua_set.shape[0]] = np.load(path)
    
np.save("dataset.npy", data)


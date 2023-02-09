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

# def find_test_idx(mus_set, tissue_mus, mus_test_idx):
#     for idx, t in enumerate(tissue_mus):
#         if idx == 0:
#             test_idx = np.where(mus_set==t)[0]
#         else:
#             test_idx = np.concatenate((test_idx,np.where(mus_set==t)[0]))
#     mus_test_idx[test_idx] += 1
#     return mus_test_idx

# mus_set = np.load(os.path.join("..","..","mcx_sim","mus_set.npy"))
# mua_set = np.load(os.path.join("..","..","mcx_sim","mua_set.npy"))

# # mus unique
# skin_mus_unique = np.unique(mus_set[:,0]) #7
# subcuit_mus_unique = np.unique(mus_set[:,1]) #7
# muscle_mus_unique = np.unique(mus_set[:,2]) #5
# ijv_mus_unique = np.unique(mus_set[:,3]) #5
# CCA_mus_unique = np.unique(mus_set[:,4]) #5

# skin_mus = np.array([skin_mus_unique[2],skin_mus_unique[4]])
# subcuit_mus = np.array([subcuit_mus_unique[2],subcuit_mus_unique[4]])
# muscle_mus = np.array([muscle_mus_unique[3]])
# ijv_mus = np.array([ijv_mus_unique[3]])
# CCA_mus = np.array([CCA_mus_unique[3]])

# skin_mus_index = [4,6] 
# subcuit_mus_index = [2,5]
# muscle_mus_index = [3]
# ijv_mus_index = [3]
# CCA_mus_index = [3]

# test_idx = np.zeros((mus_set.shape[0]))

# test_idx = find_test_idx(mus_set, skin_mus, test_idx)
# test_idx = find_test_idx(mus_set, subcuit_mus, test_idx)
# test_idx = find_test_idx(mus_set, muscle_mus, test_idx)
# test_idx = find_test_idx(mus_set, ijv_mus, test_idx)

# mus_test_idx = np.where(test_idx==4)[0]
# mus_train_idx = np.where(test_idx==0)[0]

# # mua unique
# skin_mua_unique = np.unique(mua_set[:,0]) #3
# subcuit_mua_unique = np.unique(mua_set[:,1]) #3
# muscle_mua_unique = np.unique(mua_set[:,2]) #5
# ijv_mua_unique = np.unique(mua_set[:,3]) #7
# CCA_mua_unique = np.unique(mua_set[:,4]) #7

# skin_mua = np.array([skin_mua_unique[1]])
# subcuit_mua = np.array([subcuit_mua_unique[1]])
# muscle_mua = np.array([muscle_mua_unique[2],muscle_mua_unique[4]])
# ijv_mua = np.array([ijv_mua_unique[2],ijv_mua_unique[4]])
# CCA_mua = np.array([CCA_mua_unique[2],CCA_mua_unique[4]])

# test_idx = np.zeros((mua_set.shape[0]))

# test_idx = find_test_idx(mua_set, skin_mua, test_idx)
# test_idx = find_test_idx(mua_set, subcuit_mua, test_idx)
# test_idx = find_test_idx(mua_set, muscle_mua, test_idx)
# test_idx = find_test_idx(mua_set, ijv_mua, test_idx)
# test_idx = find_test_idx(mua_set, CCA_mua, test_idx)

# mua_test_idx = np.where(test_idx==5)[0]
# mua_train_idx = np.where(test_idx==0)[0]

# test_parameter = np.concatenate([skin_mus, subcuit_mus, muscle_mus, ijv_mus, CCA_mus,
#                             skin_mua, subcuit_mua, muscle_mua, ijv_mua, CCA_mua])


# dataset_folder = os.path.join("ctchen_dataset_large")
# datapath = sorted(glob(os.path.join(dataset_folder,"*")),key=lambda x: int(x.split("_")[-1][:-4]))

# train_data = np.zeros((mus_train_idx.shape[0]*mua_train_idx.shape[0],31))
# test_data = np.zeros((mus_test_idx.shape[0]*mua_test_idx.shape[0],31))

# test_count = 0
# train_count = 0
# for idx, path in enumerate(datapath):
#     print(f"process {path}...")
#     if idx in mus_test_idx:
#         test_data[test_count*mua_test_idx.shape[0]:(test_count+1)*mua_test_idx.shape[0]] = np.load(path)[mua_test_idx]
#         test_count += 1
#     elif idx in mus_test_idx:
#         train_data[train_count*mua_train_idx.shape[0]:(train_count+1)*mua_train_idx.shape[0]] = np.load(path)[mua_train_idx]
#         train_count += 1
#     # else:
#     #     print(f"abort {path} ...")

# np.save("train_dataset.npy", train_data)
# np.save("test_dataset.npy", test_data)


# save all 

dataset_folder = os.path.join("ctchen_dataset_large")
datapath = sorted(glob(os.path.join(dataset_folder,"*")),key=lambda x: int(x.split("_")[-1][:-4]))

mus_set = np.load(os.path.join("..","..","mcx_sim","mus_set.npy"))
mua_set = np.load(os.path.join("..","..","mcx_sim","mua_set.npy"))
data = np.empty((mus_set.shape[0]*mua_set.shape[0],31))


for idx, path in enumerate(datapath):
    p = path.split("/")[-1]
    print(f"Now processing {p} .....")
    data[(idx)*mua_set.shape[0]:(idx+1)*mua_set.shape[0]] = np.load(path)
    
np.save("dataset.npy", data)


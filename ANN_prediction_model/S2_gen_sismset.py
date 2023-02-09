#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:46:46 2022

@author: md703
"""

import json
import os
import random
import numpy as np
import pandas as pd
import pickle

with open("mus_spectrum.json", "r") as f:
    mus_spectrum = json.load(f)
with open("mua_spectrum.json", "r") as f:
    mua_spectrum = json.load(f)

#%% train data
used_wl = list(np.rint(np.linspace(700, 900, 20)).astype(int))
bloodConc = [174,138]
# SO2 = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
SO2 = [i/100 for i in range(40,95,5)]
tissue = list(mus_spectrum.keys())
total_num = 10
choose_num = 3
select_spec = [i for i in range(total_num)]
random.shuffle(select_spec)

# for 1 spectrum
mus = {}
for t in tissue:
    mus[t] = pd.DataFrame(mus_spectrum[t]).to_numpy()

mua = {}
tissue = ["skin", "fat", "cca", "muscle"]
for t in tissue:
    mua[t] = pd.DataFrame(mua_spectrum[t]).to_numpy()

find_bloodConc = []
ijv_key = []
for k in list(mua_spectrum.keys()):
    if k.find("ijv") != -1:
        ijv_key.append(k)
        find_bloodConc.append(int(k.split("_")[-3]))
find_bloodConc = np.unique(np.array(find_bloodConc))

ijv = {} # size=SO2*WL  ex: 6*5
for blc in find_bloodConc:
    ijv[f'bloodConc{blc}'] = np.zeros((len(SO2),len(used_wl))) 
for blc in find_bloodConc:
    for idx,s in enumerate(SO2):
        ijv_mua = pd.DataFrame(mua_spectrum[f'ijv_bloodConc_{blc}_bloodSO2_{s}']).to_numpy()
        ijv[f'bloodConc{blc}'][idx] = ijv_mua

# muscle = {}
# muscle_SO2 = [1.0, 0.9, 0.8]
# for blc in find_bloodConc:
#     muscle[f'bloodConc{blc}'] = np.zeros((len(muscle_SO2),len(used_wl))) 
# for blc in find_bloodConc:
#     for idx,s in enumerate(muscle_SO2):
#         muscle_mua = pd.DataFrame(mua_spectrum[f'muscle_bloodConc_{blc}.0_bloodSO2_{s}']).to_numpy()
#         muscle[f'bloodConc{blc}'][idx] = muscle_mua
        
        
        
rand_choose_spec_idx = random.randint(0, total_num)
count = 0
condition = 0
ANN_input = {}
while(count < len(find_bloodConc)*len(SO2)*200000):
    print(f'process condition {condition}')
    for blc in find_bloodConc:
        if os.path.isdir(os.path.join("dataset",f"condition_{condition}")):
            os.mkdir(os.path.join("dataset",f"condition_{condition}"))
        rangdom_gen = [random.randint(0, 2*(total_num-1)),random.randint(0, 2*(total_num-1)),random.randint(0, 2*(total_num-1)),
                       random.randint(0, 2*(total_num-1)),random.randint(0, 2*(total_num-1)),random.randint(0, 2*(total_num-1)),
                       random.randint(0, 2*(total_num-1)),random.randint(0, 2*(total_num-1)),random.randint(0, 2*(total_num-1))]
        for idx,s in enumerate(SO2):
            ANN_input_dict = {"skin_mus": mus["skin"][rangdom_gen[0]],
                         "fat_mus": mus["fat"][rangdom_gen[1]],
                         "muscle_mus": mus["muscle"][rangdom_gen[2]],
                         "ijv_mus": mus["blood"][rangdom_gen[3]],
                         "cca_mus": mus["blood"][rangdom_gen[3]],
                         "skin_mua": mua["skin"][rangdom_gen[5]],
                         "fat_mua": mua["fat"][rangdom_gen[6]],
                         # "muscle_mua": muscle[f'bloodConc{blc}'][1], # SO2 = 90%
                         "muscle_mua" : mua["muscle"][rangdom_gen[7]],
                         "ijv_mua": ijv[f'bloodConc{blc}'][idx],
                         "cca_mua": mua["cca"][rangdom_gen[8]],
                         "answer": SO2[idx],
                         "bloodConc": blc}
            ANN_input_dict = pd.DataFrame(ANN_input_dict).to_numpy()
            ANN_input[f'condition_{condition}_SO2_{s}'] = ANN_input_dict
            # np.save(os.path.join("database",f"spectrum_{count}"), ANN_input)
            count += 1
        condition += 1
        
with open('large_sim_dataset.pkl', 'wb') as f:
    pickle.dump(ANN_input, f)
    
    
#%% for testing data
used_wl = list(np.rint(np.linspace(700, 900, 20)).astype(int))
bloodConc = [174,138]
# SO2 = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
SO2 = [i/100 for i in range(40,91,1)]
tissue = list(mus_spectrum.keys())
total_num = 10
choose_num = 3
select_spec = [i for i in range(total_num)]
random.shuffle(select_spec)

# for 1 spectrum
mus = {}
for t in tissue:
    mus[t] = pd.DataFrame(mus_spectrum[t]).to_numpy()

mua = {}
tissue = ["skin", "fat", "cca", "muscle"]
for t in tissue:
    mua[t] = pd.DataFrame(mua_spectrum[t]).to_numpy()

find_bloodConc = []
ijv_key = []
for k in list(mua_spectrum.keys()):
    if k.find("ijv") != -1:
        ijv_key.append(k)
        find_bloodConc.append(int(k.split("_")[-3]))
find_bloodConc = np.unique(np.array(find_bloodConc))

ijv = {} # size=SO2*WL  ex: 6*5
for blc in find_bloodConc:
    ijv[f'bloodConc{blc}'] = np.zeros((len(SO2),len(used_wl))) 
for blc in find_bloodConc:
    for idx,s in enumerate(SO2):
        ijv_mua = pd.DataFrame(mua_spectrum[f'ijv_bloodConc_{blc}_bloodSO2_{s}']).to_numpy()
        ijv[f'bloodConc{blc}'][idx] = ijv_mua

# muscle = {}
# muscle_SO2 = [1.0, 0.9, 0.8]
# for blc in find_bloodConc:
#     muscle[f'bloodConc{blc}'] = np.zeros((len(muscle_SO2),len(used_wl))) 
# for blc in find_bloodConc:
#     for idx,s in enumerate(muscle_SO2):
#         muscle_mua = pd.DataFrame(mua_spectrum[f'muscle_bloodConc_{blc}.0_bloodSO2_{s}']).to_numpy()
#         muscle[f'bloodConc{blc}'][idx] = muscle_mua
        
        
        
rand_choose_spec_idx = random.randint(0, total_num)
count = 0
condition = 0
ANN_input = {}
while(count < len(find_bloodConc)*len(SO2)*20000):
    print(f'process condition {condition}')
    for blc in find_bloodConc:
        if os.path.isdir(os.path.join("dataset",f"condition_{condition}")):
            os.mkdir(os.path.join("dataset",f"condition_{condition}"))
        rangdom_gen = [random.randint(0, 2*(total_num-1)+1),random.randint(0, 2*(total_num-1)+1),random.randint(0, 2*(total_num-1)+1),
                       random.randint(0, 2*(total_num-1)+1),random.randint(0, 2*(total_num-1)+1),random.randint(0, 2*(total_num-1)+1),
                       random.randint(0, 2*(total_num-1)+1),random.randint(0, 2*(total_num-1)+1),random.randint(0, 2*(total_num-1)+1)]
        for idx,s in enumerate(SO2):
            ANN_input_dict = {"skin_mus": mus["skin"][rangdom_gen[0]],
                         "fat_mus": mus["fat"][rangdom_gen[1]],
                         "muscle_mus": mus["muscle"][rangdom_gen[2]],
                         "ijv_mus": mus["blood"][rangdom_gen[3]], # ijv mus and cca mus should be same
                         "cca_mus": mus["blood"][rangdom_gen[3]], # ijv mus and cca mus should be same
                         "skin_mua": mua["skin"][rangdom_gen[5]],
                         "fat_mua": mua["fat"][rangdom_gen[6]],
                         # "muscle_mua": muscle[f'bloodConc{blc}'][1], # SO2 = 90%
                         "muscle_mua" : mua["muscle"][rangdom_gen[7]],
                         "ijv_mua": ijv[f'bloodConc{blc}'][idx],
                         "cca_mua": mua["cca"][rangdom_gen[8]],
                         "answer": SO2[idx],
                         "bloodConc": blc}
            ANN_input_dict = pd.DataFrame(ANN_input_dict).to_numpy()
            ANN_input[f'condition_{condition}_SO2_{s}'] = ANN_input_dict
            # np.save(os.path.join("database",f"spectrum_{count}"), ANN_input)
            count += 1
        condition += 1
        
with open('test_large_sim_dataset.pkl', 'wb') as f:
    pickle.dump(ANN_input, f)
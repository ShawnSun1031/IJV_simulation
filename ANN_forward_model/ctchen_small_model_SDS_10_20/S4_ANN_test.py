#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 23:08:47 2023

@author: md703
"""

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pickle
import seaborn as sns 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import os
#%% data preprocessing
class dataload(Dataset):
    def __init__(self, root, mus_set_path, mua_set_path):
        xy = np.load(root)
        self.mus_set = np.load(mus_set_path)
        self.mua_set = np.load(mua_set_path)
        self.x = torch.from_numpy(xy[:,:10])
        max_mus = np.max(self.mus_set, axis=0)[:5]
        max_mua = np.max(self.mua_set, axis=0)[:5]
        self.x_max = torch.from_numpy(np.concatenate((max_mus,max_mua)))
        min_mus = np.min(self.mus_set, axis=0)[:5]
        min_mua = np.min(self.mua_set, axis=0)[:5]
        self.x_min = torch.from_numpy(np.concatenate((min_mus,min_mua)))
        self.x = (self.x - self.x_min) / (self.x_max - self.x_min)
        self.y = torch.from_numpy(xy[:,11]) # SDS 14.5mm
        self.y = -torch.log(self.y)
        self.n_samples = xy.shape[0]
                
    def __getitem__(self, index):
        
        return self.x[index], self.y[index]
        
    def __len__(self):
        
        return self.n_samples
#%% model
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
            )
        
    def forward(self, x):
        return self.net(x)
    
#%%
if __name__ == "__main__":
    test_loader = torch.load("test_loader.pth")
    # plot result
    with open('trlog.pkl', 'rb') as f:
        trlog = pickle.load(f)
    min_loss = min(trlog['test_loss'])
    ep = trlog['test_loss'].index(min_loss)
    model = ANN().cuda()
    model.load_state_dict(torch.load(f"ep_{ep}_loss_{min_loss}.pth"))
    model.eval()
    
    # root = "dataset.npy"
    # xy = np.load(root)
    mus_set_path = os.path.join("..","..","mcx_sim","mus_set.npy")
    mua_set_path = os.path.join("..","..","mcx_sim","mua_set.npy")
    mus_set = np.load(mus_set_path)
    mua_set = np.load(mua_set_path)
    # x = torch.from_numpy(xy[:,:10])
    max_mus = np.max(mus_set, axis=0)[:5]
    max_mua = np.max(mua_set, axis=0)[:5]
    x_max = torch.from_numpy(np.concatenate((max_mus,max_mua)))
    min_mus = np.min(mus_set, axis=0)[:5]
    min_mua = np.min(mua_set, axis=0)[:5]
    x_min = torch.from_numpy(np.concatenate((min_mus,min_mua)))
    # x = (x - x_min) / (x_max - x_min)
    
    
    used_mus_set = mus_set[500:601]
    dataset_path = glob(os.path.join("ANN_test","ctchen_dataset_small","*"))
    error = np.zeros((101,2205,2))
    for idx,path in enumerate(dataset_path):
        data = np.load(path)
        target = data[:,[11,25]]
        data = data[:,:10]
        data = torch.from_numpy(data)
        data = (data - x_min) / (x_max - x_min)
        data = data.to(torch.float32).cuda()
        output = model(data)
        y = torch.exp(-output).detach().cpu().numpy()
        error[idx] = abs((y-target)/target)*100
        # error = error.reshape(101*2205,2)
        # np.sqrt(np.square(error.mean(0)))
    e_idx_SDS1 = np.argsort(abs(error[:,0]))
    e_idx_SDS2 = np.argsort(abs(error[:,1]))
    
    skin_mua = data[:,5]
    fat_mua = data[:,6]
    muscle_mua = data[:,7]
    ijv_mua = data[:,8]
    cca_mua = data[:,9]
    
    plt.title("ANN error test")
    plt.plot(error[:,0], "b", label="SDS1")
    plt.plot(error[:,1], "r", label="SDS2")
    # plt.plot(target[:,0], "r", label="golden")
    plt.xlabel("mus #")
    plt.ylabel("error(%)")
    plt.legend()
    plt.show()
    
    

    
    data = used_mus_set
    tissue = ['skin','fat','muscle', 'ijv', 'cca']
    for t_idx,t in enumerate(tissue):
        tissue_data = data[e_idx_SDS2,t_idx]
        if t_idx == 0:
            tissue_data_acc = (tissue_data-min_mus[t_idx])/ (max_mus[t_idx]-min_mus[t_idx])
        else:
            tissue_data_acc = (tissue_data-min_mus[t_idx])/ (max_mus[t_idx]-min_mus[t_idx])
        ax = plt.subplot()
        plt.title(f"{t}")
        plt.plot(tissue_data_acc,"o")
        plt.ylabel("level in boundary")
        
        # yticks = np.linspace(tissue_data.min(),tissue_data.max(),5)
        # xticks = np.round(error[e_idx_SDS1,0][::500], 2)
        # plt.yticks(yticks,[0,0.25,0.5,0.75,1])
        # plt.xticks([0,500,1000,1500,2000],abs(xticks))
        plt.xlabel("error")
        plt.show()
    
    plt.title("SDS2 error test")
    plt.plot(y[:,1], "b", label="test")
    plt.plot(target[:,1], "r", label="golden")
    plt.show()
    
    
    
    # error = 0
    # for batch_idx, (data,target) in enumerate(test_loader):
    #     data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
    #     output = model(data)
    #     # output = output.view(-1)
    #     y = torch.exp(-output).detach().cpu().numpy()
    #     x = torch.exp(-target).detach().cpu().numpy()

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import os

#%% data preprocessing
# class dataload(Dataset):
#     def __init__(self, root, mus_set_path, mua_set_path):
#         xy = root
#         self.mus_set = np.load(mus_set_path)
#         self.mua_set = np.load(mua_set_path)
#         self.x = torch.from_numpy(xy[:,:10])
#         max_mus = np.max(self.mus_set, axis=0)[:5]
#         max_mua = np.max(self.mua_set, axis=0)[:5]
#         self.x_max = torch.from_numpy(np.concatenate((max_mus,max_mua)))
#         min_mus = np.min(self.mus_set, axis=0)[:5]
#         min_mua = np.min(self.mua_set, axis=0)[:5]
#         self.x_min = torch.from_numpy(np.concatenate((min_mus,min_mua)))
#         self.x = (self.x - self.x_min) / (self.x_max - self.x_min)
#         self.y = torch.from_numpy(xy[:,10]) # SO2
#         self.n_samples = xy.shape[0]
                
#     def __getitem__(self, index):
        
#         return self.x[index], self.y[index]
        
#     def __len__(self):
        
#         return self.n_samples
    
#%% data preprocessing 2
def dataload(root, mus_set_path, mua_set_path):
    xy = root
    mus_set = np.load(mus_set_path)
    mua_set = np.load(mua_set_path)
    x = torch.from_numpy(xy[:,:10])
    max_mus = np.max(mus_set, axis=0)[:5]
    max_mua = np.max(mua_set, axis=0)[:5]
    x_max = torch.from_numpy(np.concatenate((max_mus,max_mua)))
    min_mus = np.min(mus_set, axis=0)[:5]
    min_mua = np.min(mua_set, axis=0)[:5]
    x_min = torch.from_numpy(np.concatenate((min_mus,min_mua)))
    x = (x - x_min) / (x_max - x_min)
    y = torch.from_numpy(xy[:,10]) # SO2
    n_samples = xy.shape[0]
    z = torch.from_numpy(xy[:,11]) # bloodConc
    
    return x,y,z




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

if __name__ == "__main__":
    condition = 200000 - 20
    used_wl = 20
    #%% small model train
    ijv_size = "small"
    SO2 = [i/100 for i in range(40,95,5)]
    with open('large_sim_dataset.pkl', 'rb') as f:
        ANN_train_input = pickle.load(f)
    prediction_ANN_output = {}
    for i in range(condition):
        for s in SO2:
            prediction_ANN_output[f'condition_{i}_SO2_{s}'] = np.zeros(41+used_wl*10+1)  #[SDS1_reflectance_wl1....wl_last, SDS2_wl1...wl_last, ans]
    model = ANN().cuda()
    model.load_state_dict(torch.load(f"{ijv_size}_ANN_model.pth"))
    count = 0
    for i in range(condition):
        print(f'processing {i}')
        for s in SO2: 
            root = ANN_train_input[f'condition_{i}_SO2_{s}']
            mus_set_path = "mus_set.npy"
            mua_set_path = "mua_set.npy"
            # dataset = dataload(root, mus_set_path, mua_set_path)
            data, target, bloodConc = dataload(root, mus_set_path, mua_set_path) # data : wl*optical_parameter = 20*10
            reflectance = model(data.to(torch.float32).cuda())
            reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][:20] = reflectance[:,0] # SDS1
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][20:40] = reflectance[:,1] # SDS2
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target.unique()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][41:61] = root[:,0] # skin mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][61:81] = root[:,1] # fat mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][81:101] = root[:,2] # muscle mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][101:121] = root[:,3] # IJV mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][121:141] = root[:,4] # CCA mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][141:161] = root[:,5] # skin mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][161:181] = root[:,6] # fat mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][181:201] = root[:,7] # muscle mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][201:221] = root[:,8] # IJV mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][221:241] = root[:,9] # CCA mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][241] = bloodConc.unique()
            
            # for wl,(data, target) in enumerate(dataset):
            #     break
            #     reflectance = model(data.to(torch.float32).cuda())
            #     reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = reflectance[0] # SDS1
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl+20] = reflectance[1] # SDS2
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = root[wl][:-1]
            #     # prediction_ANN_output['optical parameter'] = 
            #     # prediction_ANN_output[count,wl] = reflectance
            #     # prediction_ANN_output[count,-1] = target
            count += 1
    with open(f'ANN_{ijv_size}_output.pkl', 'wb') as f:
        pickle.dump(prediction_ANN_output, f)
        
    #%% small model test
    condition = 20000 - 20
    used_wl = 20
    ijv_size = "small"
    SO2 = [i/100 for i in range(40,91,1)]
    with open('test_large_sim_dataset.pkl', 'rb') as f:
        ANN_test_input = pickle.load(f)
    prediction_ANN_output = {}
    for i in range(condition):
        for s in SO2:
            prediction_ANN_output[f'condition_{i}_SO2_{s}'] = np.zeros(41+used_wl*10+1)  #[SDS1_reflectance_wl1....wl_last, SDS2_wl1...wl_last, ans]
    model = ANN().cuda()
    model.load_state_dict(torch.load(f"{ijv_size}_ANN_model.pth"))
    count = 0
    for i in range(condition):
        print(f'processing {i}')
        for s in SO2: 
            root = ANN_test_input[f'condition_{i}_SO2_{s}']
            mus_set_path = "mus_set.npy"
            mua_set_path = "mua_set.npy"
            # dataset = dataload(root, mus_set_path, mua_set_path)
            data, target, bloodConc = dataload(root, mus_set_path, mua_set_path) # data : wl*optical_parameter = 20*10
            reflectance = model(data.to(torch.float32).cuda())
            reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][:20] = reflectance[:,0] # SDS1
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][20:40] = reflectance[:,1] # SDS2
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target.unique()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][41:61] = root[:,0] # skin mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][61:81] = root[:,1] # fat mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][81:101] = root[:,2] # muscle mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][101:121] = root[:,3] # IJV mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][121:141] = root[:,4] # CCA mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][141:161] = root[:,5] # skin mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][161:181] = root[:,6] # fat mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][181:201] = root[:,7] # muscle mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][201:221] = root[:,8] # IJV mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][221:241] = root[:,9] # CCA mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][241] = bloodConc.unique()
            
            # for wl,(data, target) in enumerate(dataset):
            #     break
            #     reflectance = model(data.to(torch.float32).cuda())
            #     reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = reflectance[0] # SDS1
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl+20] = reflectance[1] # SDS2
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = root[wl][:-1]
            #     # prediction_ANN_output['optical parameter'] = 
            #     # prediction_ANN_output[count,wl] = reflectance
            #     # prediction_ANN_output[count,-1] = target
            count += 1
    with open(f'test_ANN_{ijv_size}_output.pkl', 'wb') as f:
        pickle.dump(prediction_ANN_output, f)
        
    #%% train large model
    condition = 200000 - 20
    used_wl = 20
    ijv_size = "large"
    SO2 = [i/100 for i in range(40,95,5)]
    # with open('large_sim_dataset.pkl', 'rb') as f:
    #     ANN_input = pickle.load(f)
    prediction_ANN_output = {}
    for i in range(condition):
        for s in SO2:
            prediction_ANN_output[f'condition_{i}_SO2_{s}'] = np.zeros(41+used_wl*10+1)  #[SDS1_reflectance_wl1....wl_last, SDS2_wl1...wl_last, ans]
    model = ANN().cuda()
    model.load_state_dict(torch.load(f"{ijv_size}_ANN_model.pth"))
    count = 0
    for i in range(condition):
        print(f'processing {i}')
        for s in SO2: 
            root = ANN_train_input[f'condition_{i}_SO2_{s}']
            mus_set_path = "mus_set.npy"
            mua_set_path = "mua_set.npy"
            # dataset = dataload(root, mus_set_path, mua_set_path)
            data, target, bloodConc = dataload(root, mus_set_path, mua_set_path) # data : wl*optical_parameter = 20*10
            reflectance = model(data.to(torch.float32).cuda())
            reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][:20] = reflectance[:,0] # SDS1
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][20:40] = reflectance[:,1] # SDS2
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target.unique()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][41:61] = root[:,0] # skin mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][61:81] = root[:,1] # fat mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][81:101] = root[:,2] # muscle mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][101:121] = root[:,3] # IJV mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][121:141] = root[:,4] # CCA mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][141:161] = root[:,5] # skin mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][161:181] = root[:,6] # fat mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][181:201] = root[:,7] # muscle mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][201:221] = root[:,8] # IJV mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][221:241] = root[:,9] # CCA mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][241] = bloodConc.unique()
            
            # for wl,(data, target) in enumerate(dataset):
            #     break
            #     reflectance = model(data.to(torch.float32).cuda())
            #     reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = reflectance[0] # SDS1
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl+20] = reflectance[1] # SDS2
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = root[wl][:-1]
            #     # prediction_ANN_output['optical parameter'] = 
            #     # prediction_ANN_output[count,wl] = reflectance
            #     # prediction_ANN_output[count,-1] = target
            count += 1
    with open(f'ANN_{ijv_size}_output.pkl', 'wb') as f:
        pickle.dump(prediction_ANN_output, f)
    
    #%% test large model
    condition = 20000 - 20
    used_wl = 20
    ijv_size = "large"
    SO2 = [i/100 for i in range(40,91,1)]
    # with open('test_large_sim_dataset.pkl', 'rb') as f:
    #     ANN_input = pickle.load(f)
    prediction_ANN_output = {}
    for i in range(condition):
        for s in SO2:
            prediction_ANN_output[f'condition_{i}_SO2_{s}'] = np.zeros(41+used_wl*10+1)  #[SDS1_reflectance_wl1....wl_last, SDS2_wl1...wl_last, ans]
    model = ANN().cuda()
    model.load_state_dict(torch.load(f"{ijv_size}_ANN_model.pth"))
    count = 0
    for i in range(condition):
        print(f'processing {i}')
        for s in SO2: 
            root = ANN_test_input[f'condition_{i}_SO2_{s}']
            mus_set_path = "mus_set.npy"
            mua_set_path = "mua_set.npy"
            # dataset = dataload(root, mus_set_path, mua_set_path)
            data, target, bloodConc = dataload(root, mus_set_path, mua_set_path) # data : wl*optical_parameter = 20*10
            reflectance = model(data.to(torch.float32).cuda())
            reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][:20] = reflectance[:,0] # SDS1
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][20:40] = reflectance[:,1] # SDS2
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target.unique()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][41:61] = root[:,0] # skin mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][61:81] = root[:,1] # fat mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][81:101] = root[:,2] # muscle mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][101:121] = root[:,3] # IJV mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][121:141] = root[:,4] # CCA mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][141:161] = root[:,5] # skin mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][161:181] = root[:,6] # fat mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][181:201] = root[:,7] # muscle mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][201:221] = root[:,8] # IJV mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][221:241] = root[:,9] # CCA mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][241] = bloodConc.unique()
            
            # for wl,(data, target) in enumerate(dataset):
            #     break
            #     reflectance = model(data.to(torch.float32).cuda())
            #     reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = reflectance[0] # SDS1
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl+20] = reflectance[1] # SDS2
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = root[wl][:-1]
            #     # prediction_ANN_output['optical parameter'] = 
            #     # prediction_ANN_output[count,wl] = reflectance
            #     # prediction_ANN_output[count,-1] = target
            count += 1
    with open(f'test_ANN_{ijv_size}_output.pkl', 'wb') as f:
        pickle.dump(prediction_ANN_output, f)
    
    
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pickle
import seaborn as sns 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
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

#%% plot y=x
def plot_line():
    model.eval()
    error = 0
    for batch_idx, (data,target) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        # output = output.view(-1)
        y = torch.exp(-output).detach().cpu().numpy()
        x = torch.exp(-target).detach().cpu().numpy()
        error += torch.sqrt(torch.square((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).mean()).item()
        if batch_idx == 0:
            plt.plot(x,y, 'r.', markersize=5, label='predict')
            plt.plot(x,x,'b', label='optimal')
        else:
            plt.plot(x,y, 'r.', markersize=5)
            plt.plot(x,x,'b')
    error = error/len(test_loader)
    plt.title(f"RMSPE:{100*error:.2f}%")
    plt.xlabel("truth reflectance")
    plt.ylabel("predict reflectance")
    # plt.legend()
    # plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig("RMSPE_all.png")
    plt.show()
    
    # plot individual
    error = 0
    for batch_idx, (data,target) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        output = output[:,0].view(-1)
        y = torch.exp(-output).detach().cpu().numpy()
        x = torch.exp(-target[:,0]).detach().cpu().numpy()
        error += torch.sqrt(torch.square((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).mean()).item()
        plt.plot(x,y, 'r.', markersize=5)
        plt.plot(x,x,'b')
    error = error/len(test_loader)
    plt.title(f"RMSPE:{100*error:.2f}%")
    plt.xlabel("truth reflectance")
    plt.ylabel("predict reflectance")
    plt.savefig("RMSPE_SDS_10.png")
    plt.show()
    
    error = 0
    for batch_idx, (data,target) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        output = output[:,-1].view(-1)
        y = torch.exp(-output).detach().cpu().numpy()
        x = torch.exp(-target[:,-1]).detach().cpu().numpy()
        error += torch.sqrt(torch.square((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).mean()).item()
        plt.plot(x,y, 'r.', markersize=5)
        plt.plot(x,x,'b')
    error = error/len(test_loader)
    plt.title(f"RMSPE:{100*error:.2f}%")
    plt.xlabel("truth reflectance")
    plt.ylabel("predict reflectance")
    plt.savefig("RMSPE_SDS_20.png")
    plt.show()
    
#%% plot hist
def plot_hist():
    model.eval()
    error = 0
    error_set = {"error":[]}
    for batch_idx, (data,target) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        # output = output.view(-1)
        y = torch.exp(-output).detach().cpu().numpy()
        x = torch.exp(-target).detach().cpu().numpy()
        error += torch.sqrt(torch.square((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).mean()).item()
        e = 100*((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).mean(-1).numpy()
        for i in e:
            error_set['error'].append(i)
    error = error/len(test_loader)        
    error_set = pd.DataFrame(error_set)
    plt.figure(figsize=(12,6))
    s = sns.histplot(data=error_set, x='error')
    std = np.std(error_set).to_numpy()[0]
    mean = np.mean(error_set).to_numpy()[0]
    plt.axvline(mean, *s.get_ylim(),color='b')
    plt.axvline(mean+2*std, *s.get_ylim(),color='r')
    plt.axvline(mean-2*std, *s.get_ylim(),color='r')
    plt.text(mean,s.get_ylim()[0],f"{mean:.2f}%",ha='center',va='bottom')
    plt.text(mean+2*std,s.get_ylim()[0],f"{mean+2*std:.2f}%",ha='center',va='bottom')
    plt.text(mean-2*std,s.get_ylim()[0],f"{mean-2*std:.2f}%",ha='center',va='bottom')
    plt.title(f"testing error histogram  \n mean error: {mean:.2f}% std error: {std:.2f}% RMSPE: {100*error:.2f}%")
    plt.savefig("error_hist_all.png")
    plt.show()
    
    error = 0
    error_set = {"error":[]}
    for batch_idx, (data,target) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        output = output[:,0].view(-1)
        y = torch.exp(-output).detach().cpu().numpy()
        x = torch.exp(-target[:,0]).detach().cpu().numpy()
        error += torch.sqrt(torch.square((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).mean()).item()
        e = 100*((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).numpy()
        for i in e:
            error_set['error'].append(i)
    error = error/len(test_loader)        
    error_set = pd.DataFrame(error_set)
    plt.figure(figsize=(12,6))
    s = sns.histplot(data=error_set, x='error')
    std = np.std(error_set).to_numpy()[0]
    mean = np.mean(error_set).to_numpy()[0]
    plt.axvline(mean, *s.get_ylim(),color='b')
    plt.axvline(mean+2*std, *s.get_ylim(),color='r')
    plt.axvline(mean-2*std, *s.get_ylim(),color='r')
    plt.text(mean,s.get_ylim()[0],f"{mean:.2f}%",ha='center',va='bottom')
    plt.text(mean+2*std,s.get_ylim()[0],f"{mean+2*std:.2f}%",ha='center',va='bottom')
    plt.text(mean-2*std,s.get_ylim()[0],f"{mean-2*std:.2f}%",ha='center',va='bottom')
    plt.title(f"testing error histogram  \n mean error: {mean:.2f}% std error: {std:.2f}% RMSPE: {100*error:.2f}%")
    plt.savefig("error_hist_SDS_10.png")
    plt.show()
    
    error = 0
    error_set = {"error":[]}
    for batch_idx, (data,target) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        output = output[:,-1].view(-1)
        y = torch.exp(-output).detach().cpu().numpy()
        x = torch.exp(-target[:,-1]).detach().cpu().numpy()
        error += torch.sqrt(torch.square((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).mean()).item()
        e = 100*((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).numpy()
        for i in e:
            error_set['error'].append(i)
    error = error/len(test_loader)        
    error_set = pd.DataFrame(error_set)
    plt.figure(figsize=(12,6))
    s = sns.histplot(data=error_set, x='error')
    std = np.std(error_set).to_numpy()[0]
    mean = np.mean(error_set).to_numpy()[0]
    plt.axvline(mean, *s.get_ylim(),color='b')
    plt.axvline(mean+2*std, *s.get_ylim(),color='r')
    plt.axvline(mean-2*std, *s.get_ylim(),color='r')
    plt.text(mean,s.get_ylim()[0],f"{mean:.2f}%",ha='center',va='bottom')
    plt.text(mean+2*std,s.get_ylim()[0],f"{mean+2*std:.2f}%",ha='center',va='bottom')
    plt.text(mean-2*std,s.get_ylim()[0],f"{mean-2*std:.2f}%",ha='center',va='bottom')
    plt.title(f"testing error histogram  \n mean error: {mean:.2f}% std error: {std:.2f}% RMSPE: {100*error:.2f}%")
    plt.savefig("error_hist_SDS_20.png")
    plt.show()
    
    error = 0
    error_set = {"SDS10_error":[]}
    for batch_idx, (data,target) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        output = output[:,0].view(-1)
        y = torch.exp(-output).detach().cpu().numpy()
        x = torch.exp(-target[:,0]).detach().cpu().numpy()
        error += torch.sqrt(torch.square((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).mean()).item()
        e = 100*((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).numpy()
        for i in e:
            error_set['SDS10_error'].append(i)
    error = error/len(test_loader)        
    error_set = pd.DataFrame(error_set)
    plt.figure(figsize=(12,6))
    s = sns.histplot(data=error_set, x='SDS10_error', color="red")
    
    error = 0
    error_set = {"SDS20_error":[]}
    for batch_idx, (data,target) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        output = output[:,-1].view(-1)
        y = torch.exp(-output).detach().cpu().numpy()
        x = torch.exp(-target[:,-1]).detach().cpu().numpy()
        error += torch.sqrt(torch.square((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).mean()).item()
        e = 100*((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).numpy()
        for i in e:
            error_set['SDS20_error'].append(i)
    error = error/len(test_loader)        
    error_set = pd.DataFrame(error_set)
    s = sns.histplot(data=error_set, x='SDS20_error')
    plt.legend(['SDS10', 'SDS20'])
    plt.savefig("error_hist_merge.png")
    plt.show()

#%% plot train test loss
def plot_loss(trlog):
    tr_loss = trlog['train_loss'][5:-1]
    ts_loss = trlog['test_loss'][5:-1]
    epoch = range(len(tr_loss))
    plt.plot(epoch,tr_loss,'blue')
    plt.plot(epoch,ts_loss,'r')
    plt.legend(["train loss", "test loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("loss.png")
    plt.show()
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
    # plot_loss(trlog)
    plot_line()
    # plot_hist()
    
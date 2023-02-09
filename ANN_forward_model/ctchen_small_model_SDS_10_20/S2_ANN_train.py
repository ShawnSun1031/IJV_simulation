import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
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
        self.y = torch.from_numpy(xy[:,[11,25]]) # SDS2:10.09mm  SDS16: 20.38mm
        self.y = -torch.log(self.y)
        self.n_samples = xy.shape[0]
                
    def __getitem__(self, index):
        
        return self.x[index], self.y[index]
        
    def __len__(self):
        
        return self.n_samples

def data_preprocess(dataset, batch_size, test_split, shuffle_dataset, random_seed):
    # create data indice for training and testing splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    # count out split size
    split = int(np.floor(test_split*dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:],indices[:split]

    # creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, test_loader

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
    
#%% train model
def train():
    trlog = {}
    trlog['train_loss'] = []
    trlog['test_loss'] = []
    min_loss = 100000
    for ep in range(epoch):
        model.train()
        tr_loss = 0
        for batch_idx, (data,target) in enumerate(train_loader):
            data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
            optimizer.zero_grad()
            output = model(data)
            # target = target.view(-1,1)
            # loss = torch.sqrt(torch.square((output-target)/target).mean())
            loss = criterion(output,target)
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % int(0.1*len(train_loader)) == 0:
                print(f"[train] ep:{ep}/{epoch}({100*ep/epoch:.2f}%) batch:{batch_idx}/{len(train_loader)}({100*batch_idx/len(train_loader):.2f}%)\
                      loss={tr_loss/(batch_idx+1)}")
        trlog['train_loss'].append(tr_loss/len(train_loader))
        min_loss = test(trlog,ep,min_loss)
        
    
    return trlog

def test(trlog,ep,min_loss):
    model.eval()
    ts_loss = 0
    for batch_idx, (data,target) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        optimizer.zero_grad()
        output = model(data)
        # target = target.view(-1,1)
        # loss = torch.sqrt(torch.square((output-target)/target).mean())
        loss = criterion(output,target)
        ts_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    print(f"[test] batch:{batch_idx}/{len(test_loader)}({100*batch_idx/len(test_loader):.2f}%) loss={ts_loss/len(test_loader)}")
    trlog['test_loss'].append(ts_loss/len(test_loader))
    
    if min_loss > ts_loss/len(test_loader):
        min_loss = ts_loss/len(test_loader)
        torch.save(model.state_dict(), f"ep_{ep}_loss_{min_loss}.pth")
        with open('trlog.pkl', 'wb') as f:
            pickle.dump(trlog, f)
            
    return min_loss

#%% plot pred
def pred():
    model.eval()
    error = 0
    for batch_idx, (data,target) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        optimizer.zero_grad()
        output = model(data)
        # output = output.view(-1)
        y = torch.exp(-output).detach().cpu().numpy()
        x = torch.exp(-target).detach().cpu().numpy()
        error += torch.sqrt(torch.square((torch.tensor(y)-torch.tensor(x))/torch.tensor(x)).mean()).item()
        plt.plot(x,y, 'r.', markersize=5)
        plt.plot(x,x,'b')
    error = error/len(test_loader)
    plt.title(f"RMSPE {100*error:.2f}%")
    plt.xlabel("truth reflectance")
    plt.ylabel("predict reflectance")
    plt.savefig("RMSPE.png")
    plt.show()
    
#%%   
if __name__ == "__main__":
    root = "dataset.npy"
    mus_set_path = os.path.join("..","..","mcx_sim","mus_set.npy")
    mua_set_path = os.path.join("..","..","mcx_sim","mua_set.npy")
    batch_size = 128
    # split data setting
    # set testing data size
    test_split = 0.2
    # need shuffle or not
    shuffle_dataset = True
    # random seed of shuffle 
    random_seed = 703
    dataset = dataload(root,mus_set_path,mua_set_path)
    # train_dataset = dataload(root="train_dataset.npy")
    # test_dataset = dataload(root="test_dataset.npy")
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader, test_loader = data_preprocess(dataset, batch_size, test_split, shuffle_dataset, random_seed)
    torch.save(train_loader, "train_loader.pth")
    torch.save(test_loader, "test_loader.pth")
    
    # train model
    model = ANN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    epoch = 500
    trlog = train()
    with open('trlog.pkl', 'wb') as f:
        pickle.dump(trlog, f)
    
    # plot result
    with open('trlog.pkl', 'rb') as f:
        trlog = pickle.load(f)
    min_loss = min(trlog['test_loss'])
    ep = trlog['test_loss'].index(min_loss)
    model = ANN().cuda()
    model.load_state_dict(torch.load(f"ep_{ep}_loss_{min_loss}.pth"))
    pred()
    
    
        
    
    
    
    
    

from random import seed, shuffle
from tqdm import tqdm
from lib.env import *
import os
import torch

def get_ids():
    return sorted([id.split("_")[0] for id in os.listdir(DATASET_PATH) if "PF" in id])

def get_leave_one_out_cv_ids_for_ekyn():
    ids = get_ids()
    seed(0)
    shuffle(ids)
    ret = []
    for test_id in ids:
        train_ids = [x for x in ids if x != test_id]
        ret.append((train_ids, [test_id]))
    return ret
def get_dataloaders_for_fold(fold):
    folds = get_leave_one_out_cv_ids_for_ekyn()
    train_ids,test_ids = folds[fold]

    subjects = [load_eeg_label_pair(id=id,condition=condition,zero_pad=True) for id in train_ids for condition in ['Vehicle','PF']]
    Xs = [subject[0] for subject in subjects]
    ys = [subject[1] for subject in subjects]

    print(len(Xs),len(ys))
    X = torch.cat(Xs)
    print(X.shape)

    y = []

    for i,yi in enumerate(ys):
        first_value = yi[0]
        last_value = yi[-1]

        pad_left = first_value.repeat(4,1)
        pad_right = last_value.repeat(4,1)

        if i == 0:
            yi = torch.cat([yi, pad_right])
        elif i == len(ys)-1:
            yi = torch.cat([pad_left,yi])
        else:
            yi = torch.cat([pad_left, yi, pad_right])

        y.append(yi)
    y = torch.cat(y)

    class SSDataset(torch.utils.data.Dataset):
        def __init__(self,X,y,idx) -> None:
            super().__init__()
            self.X = X
            self.y = y
            self.idx = idx
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, index):
            i = self.idx[index]
            return (self.X[i:i+9].flatten(),self.y[i])
    from sklearn.model_selection import train_test_split
    train_idx,dev_idx = train_test_split(range(len(y)),test_size=.1,random_state=0,shuffle=True)
    from torch.utils.data import DataLoader
    trainloader = DataLoader(dataset=SSDataset(X,y,train_idx),batch_size=128,shuffle=True)
    devloader = DataLoader(dataset=SSDataset(X,y,dev_idx),batch_size=32,shuffle=True)
    
    return trainloader,devloader
def load_eeg_label_pair(id,condition,zero_pad):
    X,y = torch.load(f'{DATASET_PATH}/{id}_{condition}.pt',weights_only=False)
    if zero_pad:
        X = torch.cat([torch.zeros(WINDOW_SIZE//2,5000),X,torch.zeros(WINDOW_SIZE//2,5000)])
    return (X,y)

class SSDataset(torch.utils.data.Dataset):
    def __init__(self,Xs,ys,idx) -> None:
        super().__init__()
        self.Xs = Xs
        self.ys = ys
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, index):
        index = self.idx[index]
        return (self.Xs[index // 8640][(index % 8640) : (index % 8640) + 9].flatten(),self.ys[index // 8640][index % 8640])

class Windowset(torch.utils.data.Dataset):
    def __init__(self,X,y):
        self.X = torch.cat([torch.zeros(4,5000),X,torch.zeros(4,5000)])
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx:idx+9].flatten(),self.y[idx])
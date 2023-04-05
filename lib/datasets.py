from os.path import join
from torch.utils.data import Dataset
from torch import load
import torch
class EEGDataset(Dataset):
    def __init__(self,dir,labels):
        self.labels = load(f'{labels}')
        self.dir = dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = join(self.dir, str(idx)+".pt")
        X = load(path).reshape(-1,5000)[:,::10]
        y = self.labels[idx]

        return (X,y)
class WindowedEEGDataset(Dataset):
    def __init__(self,dir,ids):
        self.dir = dir
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        X,y = torch.load(f'{self.dir}/{id}.pt')

        return (X,y)
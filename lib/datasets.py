import os
from torch.utils.data import Dataset,DataLoader
import torch
import os
import pandas as pd
from torch.utils.data import Dataset,DataLoader

class EEGDataset(Dataset):
    def __init__(self,dir,labels):
        self.labels = torch.load(f'{labels}')
        self.dir = dir

    def __len__(self):
        return int(len(self.labels)/10)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, str(idx)+".pt")
        X = torch.load(path).reshape(-1,5000)
        y = self.labels[idx]

        return (X,y)

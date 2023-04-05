from lib.utils import load_raw_list
import pandas as pd
from tqdm import tqdm
import torch
import os

from torch import nn
from torch.nn.functional import relu,one_hot
from sklearn.model_selection import train_test_split
from lib.datasets import WindowedEEGDataset
from torch.utils.data import TensorDataset,DataLoader
from lib.utils import *

device = 'cuda'

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(15000,128)
        self.fc2 = nn.Linear(128,3)
    def forward(self,x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x

        
model = torch.load('model_1200.pt').cpu()
data_dir = 'windowsize_3'

train_idx,dev_idx = train_test_split(range(len(os.listdir(data_dir))),test_size=.2,shuffle=True,random_state=0)

devloader = DataLoader(WindowedEEGDataset(data_dir,dev_idx),batch_size=128,shuffle=True)
# # test confusion matrices
y_true = torch.Tensor()
y_pred = torch.Tensor()
for (X,y) in tqdm(devloader):
    y_true = torch.cat([y_true,y.argmax(axis=1)])
    y_pred = torch.cat([y_pred,torch.softmax(model(X),dim=1).argmax(axis=1)])
y_pred = y_pred
cms(y_true=y_true,y_pred=y_pred,current_date=None)
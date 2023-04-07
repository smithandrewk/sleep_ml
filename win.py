from lib.utils import load_raw_list
import pandas as pd
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt

from torch import nn
from torch.nn.functional import relu,one_hot
from sklearn.model_selection import train_test_split
from lib.datasets import WindowedEEGDataset
from torch.utils.data import DataLoader

device = 'cuda'
data_dir = 'data/w1'
project_dir = 'w1'

train_idx,dev_idx = train_test_split(range(len(os.listdir(data_dir))),test_size=.2,shuffle=True,random_state=0)

trainloader = DataLoader(WindowedEEGDataset(data_dir,train_idx),batch_size=128,shuffle=True)
devloader = DataLoader(WindowedEEGDataset(data_dir,dev_idx),batch_size=128,shuffle=True)

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5000,32)
        self.fc2 = nn.Linear(32,3)
    def forward(self,x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x
    
model = MLP().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)

loss_tr = []
loss_dev = []

for i in tqdm(range(3)):
    # train loop
    model.train()
    loss_tr_total = 0
    for (X_tr,y_tr) in trainloader:
        X_tr,y_tr = X_tr.to(device),y_tr.to(device)
        logits = model(X_tr)
        loss = criterion(logits,y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tr_total += loss.item()
    loss_tr.append(loss_tr_total/len(trainloader))
    # dev loop
    model.eval()
    loss_dev_total = 0
    for (X_dv,y_dv) in devloader:
        X_dv,y_dv = X_dv.to(device),y_dv.to(device)
        logits = model(X_dv)
        loss = criterion(logits,y_dv)
        loss_dev_total += loss.item()
    loss_dev.append(loss_dev_total/len(devloader))

    # plot recent loss
    plt.plot(loss_tr[-30:])
    plt.plot(loss_dev[-30:])
    plt.savefig(f'{project_dir}/loss.jpg')
    plt.close()

    # plot all loss
    plt.plot(loss_tr)
    plt.plot(loss_dev)
    plt.savefig(f'{project_dir}/curr_loss.jpg')
    plt.close()

    # print stats
    print(f'Epoch {i} Train: {loss_tr_total/len(trainloader)} Dev: {loss_dev_total/len(devloader)}')

    # save on checkpoint
    if(i%10 == 0):
        torch.save(model,f'{project_dir}/{i}.pt')


torch.save(model,f'{project_dir}/model.pt')
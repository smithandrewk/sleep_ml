"""
author: Andrew Smith
date: Mar 21
description:
"""
import pandas as pd
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import json

from lib.utils import *
from lib.models import CNN as MODEL
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from lib.datasets import Dataset2p0
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

data_dir = 'w1_small_balanced_normalized'

# argparse
parser = argparse.ArgumentParser(description='Training program')
parser.add_argument('-r','--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument("-e", "--epochs", type=int, default=100,help="Number of training iterations")
parser.add_argument("-d", "--device", type=int, default=0,help="Cuda device to select")
parser.add_argument("-b", "--batch", type=int, default=64,help="Batch Size")
args = parser.parse_args()

current_date = str(datetime.now()).replace(' ','_')
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
config = {
    'BATCH_SIZE':args.batch,
    'EPOCHS':args.epochs,
    'RESUME':args.resume,
    'START_TIME':current_date
}

if not os.path.isdir(f'project'):
    os.system(f'mkdir project')
if not os.path.isdir(f'project/{current_date}'):
    os.system(f'mkdir project/{current_date}')
project_dir = 'project'

# train_idx,dev_idx = train_test_split(range(len(os.listdir(data_dir))),test_size=.2,shuffle=True,random_state=0)

# trainloader = DataLoader(WindowedEEGDataset(data_dir,train_idx),batch_size=128,shuffle=True)
# devloader = DataLoader(WindowedEEGDataset(data_dir,dev_idx),batch_size=128,shuffle=True)

trainloader = DataLoader(Dataset2p0(dir=f'{data_dir}/train/',labels=f'{data_dir}/y_train.pt'),batch_size=64,shuffle=True)
devloader = DataLoader(Dataset2p0(dir=f'{data_dir}/test/',labels=f'{data_dir}/y_test.pt'),batch_size=64,shuffle=True)

model = MODEL()

if(config['RESUME']):
    print("Resuming previous training")
    if os.path.exists(f'project/model.pt'):
        model.load_state_dict(torch.load(f='project/model.pt'))
    else:
        print("Model file does not exist.")
        print("Exiting because resume flag was given and model does not exist. Either remove resume flag or move model to directory.")
        exit(0)
    with open(f'project/config.json','r') as f:
        previous_config = json.load(f)
    config['START_EPOCH'] = previous_config['END_EPOCH'] + 1
else:
    config['START_EPOCH'] = 0

config['END_EPOCH'] = config['START_EPOCH'] + config['EPOCHS'] - 1

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)

loss_tr = []
loss_dev = []

pbar = tqdm(range(config['EPOCHS']))

for epoch in pbar:
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

    pbar.set_description(f'\033[94mDev Loss: {loss_tr[-1]:.4f}\033[93m Val Loss: {loss_dev[-1]:.4f}\033[0m')

    # plot recent loss
    plt.plot(loss_tr[-30:])
    plt.plot(loss_dev[-30:])
    plt.savefig(f'{project_dir}/{current_date}/loss.jpg')
    plt.close()

    # plot all loss
    plt.plot(loss_tr)
    plt.plot(loss_dev)
    plt.savefig(f'{project_dir}/{current_date}/curr_loss.jpg')
    plt.close()

    # save on checkpoint
    torch.save(model,f'{project_dir}/{current_date}/{epoch}.pt')

# test confusion matrices
y_true = torch.Tensor()
y_pred = torch.Tensor().cuda()
for (X,y) in devloader:
    y_true = torch.cat([y_true,y.argmax(axis=1)])
    y_pred = torch.cat([y_pred,softmax(model(X.cuda()),dim=1).argmax(axis=1)])
y_pred = y_pred.cpu()
cms(y_true=y_true,y_pred=y_pred,current_date=current_date)

# save model
if torch.cuda.device_count() > 1:
    torch.save(model.module.state_dict(), f=f'project/{current_date}/model.pt')
    torch.save(model.module.state_dict(), f=f'project/model.pt')
else:
    torch.save(model.state_dict(), f=f'project/{current_date}/model.pt')
    torch.save(model.state_dict(), f=f'project/model.pt')

# save config
with open('project/config.json', 'w') as f:
     f.write(json.dumps(config))
with open(f'project/{current_date}/config.json', 'w') as f:
     f.write(json.dumps(config))
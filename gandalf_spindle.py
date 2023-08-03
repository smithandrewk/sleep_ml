#! /usr/bin/env python3
"""
author: Andrew Smith
date: Mar 21
description:
"""
import json
import argparse
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from lib.utils import *
from lib.models import Frodo
from lib.ekyn import *
from lib.datasets import *
from mne.io import read_raw_edf
from scipy.signal import resample
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def load_spindle_eeg_label_pair(cohort='A',subject='1'):
    if cohort == 'C':
        fs = 200
    else:
        fs = 128
    raw = read_raw_edf(f'./data/Cohort{cohort}/recordings/{cohort}{subject}.edf')
    eeg = raw.get_data('EEG1').squeeze()
    eeg = resample(eeg,86400*500)
    X = torch.from_numpy(eeg.reshape(-1,5000)).float()
    df = pd.read_csv(f'data/Cohort{cohort}/scorings/{cohort}{subject}.csv',header=None)
    cat = pd.Categorical(df[1])
    cats = cat.categories
    labels = np.array([[a]*2000 for a in list(cat.codes)]).flatten()
    y = torch.from_numpy(labels.reshape(-1,5000)).mode(dim=1).values
    if f'{cohort}{subject}' in ['D1','D2','D3','C1','C2','C3','C4','C5','C6']:
        # ['1', 'n', 'r', 'w']
        y[torch.where(y == 0)[0]] = 3
        y[torch.where(y == 2)[0]] = 0
        y[torch.where(y == 3)[0]] = 2
    elif f'{cohort}{subject}' in ['D4','D5','D6']:
        # ['n', 'r', 'w']
        y[torch.where(y == 1)[0]] = 3
        y[torch.where(y == 0)[0]] = 1
        y[torch.where(y == 3)[0]] = 0
    elif f'{cohort}{subject}' in ['A2','B1']:
        # ['1', '2', '3', 'a', 'n', 'r', 'w']
        y[torch.where(y == 0)[0]] = 6
        y[torch.where(y == 1)[0]] = 4
        y[torch.where(y == 2)[0]] = 5
        y[torch.where(y == 3)[0]] = 5
        y[torch.where(y == 4)[0]] = 1
        y[torch.where(y == 5)[0]] = 0
        y[torch.where(y == 6)[0]] = 2
    else:
        # ['1', '2', '3', 'n', 'r', 'w']
        y[torch.where(y == 0)[0]] = 5
        y[torch.where(y == 1)[0]] = 3
        y[torch.where(y == 2)[0]] = 4
        y[torch.where(y == 3)[0]] = 1
        y[torch.where(y == 4)[0]] = 0
        y[torch.where(y == 5)[0]] = 2
    y = torch.nn.functional.one_hot(y.long()).float()
    X = torch.cat([zeros(9//2,5000),X,zeros(9//2,5000)])
    return X,y
# argparse
parser = argparse.ArgumentParser(description='Training program')
parser.add_argument('-r','--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument("-e", "--epochs", type=int, default=1000,help="Number of training iterations")
parser.add_argument("-d", "--device", type=int, default=0,help="Cuda device to select")
parser.add_argument("-f", "--fold", type=str, default=0,help="Fold from 0-15")
args = parser.parse_args()

FOLD = int(args.fold)
current_date = str(datetime.datetime.now()).replace(' ','_')
project_dir = f'gandalf_spindle'
PATIENCE = 30
lr = 3e-4
batch_size = 32
DEVICE = f'cuda:{args.device}' if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"

print(f'device: {DEVICE}')

class Gandalf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Frodo(n_features=5000,device=DEVICE).to(DEVICE)
        self.lstm = nn.LSTM(16,32,bidirectional=True)
        self.fc1 = nn.Linear(64,3)
    def forward(self,x_2d,classification=True):
        x_2d = x_2d.view(-1,9,1,5000)
        x = torch.Tensor().to(DEVICE)
        for t in range(x_2d.size(1)):
            xi = self.encoder(x_2d[:,t,:,:],classification=False)
            x = torch.cat([x,xi.unsqueeze(0)],dim=0)
        out,_ = self.lstm(x)
        if(classification):
            x = self.fc1(out[-1])
        else:
            x = out[-1]
        return x
model = Gandalf()

config = {
    'MODEL':str(model),
    'BATCH_SIZE':batch_size,
    'EPOCHS':args.epochs,
    'RESUME':args.resume,
    'START_TIME':current_date,
    'LEARNING_RATE':lr,
    'PATIENCE':PATIENCE,
    'BEST_DEV_LOSS':torch.inf,
    'START_EPOCH':0,
    'BEST_MODEL_EPOCH':0
}

if not os.path.isdir(project_dir):
    os.system(f'mkdir {project_dir}')
if not os.path.isdir(f'{project_dir}/{current_date}'):
    os.system(f'mkdir {project_dir}/{current_date}')

ids = ['A1','A2','A3','A4','B1','B2','B3','B4','C1','C2','C3','C4','C5','C6','C7','C8']
subjects = [load_spindle_eeg_label_pair(cohort=id[0],subject=id[1]) for id in ids]
Xs = [subject[0] for subject in subjects]
ys = [subject[1] for subject in subjects]
train_idx,test_idx = train_test_split(range(len(subjects)*8640),test_size=.1,random_state=0,shuffle=True)
trainloader = DataLoader(dataset=SSDataset(Xs,ys,train_idx),batch_size=32,shuffle=True)
devloader = DataLoader(dataset=SSDataset(Xs,ys,test_idx),batch_size=32,shuffle=True)
print(f'trainloader: {len(trainloader)} batches')
print(f'devloader: {len(devloader)} batches')

params = sum([p.flatten().size()[0] for p in list(model.parameters())])
print("Params: ",params)

if(config['RESUME']):
    print("Resuming previous training")
    if os.path.exists(f'{project_dir}/last_model.pt'):
        model.load_state_dict(torch.load(f=f'{project_dir}/last_model.pt',map_location='cpu'))
    else:
        print("Model file does not exist.")
        print("Exiting because resume flag was given and model does not exist. Either remove resume flag or move model to directory.")
        exit(0)
    with open(f'{project_dir}/config.json','r') as f:
        previous_config = json.load(f)
    config['START_EPOCH'] = previous_config['END_EPOCH'] + 1
    config['BEST_DEV_LOSS'] = previous_config['BEST_DEV_LOSS']
    config['BEST_MODEL_EPOCH'] = previous_config['BEST_MODEL_EPOCH']

config['END_EPOCH'] = config['START_EPOCH'] + config['EPOCHS'] - 1

model.to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

loss_tr = []
loss_dev = []

unimproved_epochs = 0

pbar = tqdm(range(config['EPOCHS']))

for epoch in pbar:
    loss_tr.append(training_loop(model,trainloader,criterion,optimizer,DEVICE))
    loss_dev.append(development_loop(model,devloader,criterion,DEVICE))

    if (PATIENCE != None):
        if (loss_dev[-1] < config['BEST_DEV_LOSS']):
            config['BEST_DEV_LOSS'] = loss_dev[-1]
            config['BEST_MODEL_EPOCH'] = epoch
            unimproved_epochs = 0
            torch.save(model.state_dict(), f=f'{project_dir}/best_model.pt')
        else:
            unimproved_epochs += 1
            if (unimproved_epochs >= PATIENCE): # early stopping
                break

    pbar.set_description(f'\033[94m Train Loss: {loss_tr[-1]:.4f}\033[93m Dev Loss: {loss_dev[-1]:.4f}\033[92m Best Loss: {config["BEST_DEV_LOSS"]:.4f} \033[91m Stopping: {unimproved_epochs}\033[0m')


    # plot recent loss
    plt.figure()
    plt.plot(loss_tr[-30:])
    plt.plot(loss_dev[-30:])
    plt.savefig(f'{project_dir}/{current_date}/loss_last_30.jpg')
    plt.close()

    # plot all loss
    plt.plot(loss_tr)
    plt.plot(loss_dev)
    plt.savefig(f'{project_dir}/{current_date}/loss_all_epochs.jpg')
    plt.close()

    # save on checkpoint
    torch.save(model.state_dict(), f=f'{project_dir}/{current_date}/{epoch}.pt')

_,_,y_true,y_pred,_ = evaluate(devloader,model,criterion,DEVICE)
cm_grid(y_true=y_true,y_pred=y_pred,save_path=f'{project_dir}/{current_date}/cm_last_dev.jpg')

torch.save(model.state_dict(), f=f'{project_dir}/{current_date}/last_model.pt')
torch.save(model.state_dict(), f=f'{project_dir}/last_model.pt')

# save config
with open(f'{project_dir}/config.json', 'w') as f:
     f.write(json.dumps(config))
with open(f'{project_dir}/{current_date}/config.json', 'w') as f:
     f.write(json.dumps(config))
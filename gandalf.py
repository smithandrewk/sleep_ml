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
from torch import nn
from torch.nn.functional import relu

from lib.utils import *
from lib.ekyn import *
from lib.models import Gandalf

parser = argparse.ArgumentParser(description='Training program')
parser.add_argument('-r','--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument("-e", "--epochs", type=int, default=1000,help="Number of training iterations")
parser.add_argument("-d", "--device", type=int, default=0,help="Cuda device to select")
parser.add_argument("-f", "--fold", type=str, default=0,help="Fold from 0-15")
args = parser.parse_args()

FOLD = int(args.fold)

model = Gandalf()

config = {
    'MODEL':str(model),
    'BATCH_SIZE':batch_size,
    'EPOCHS':args.epochs,
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

folds = get_leave_one_out_cv_ids_for_ekyn()
train_ids,test_ids = folds[FOLD]
subjects = [load_eeg_label_pair(id=id,condition=condition,zero_pad=True,windowsize=9) for id in train_ids for condition in ['Vehicle','PF']]
Xs = [subject[0] for subject in subjects]
ys = [subject[1] for subject in subjects]
train_idx,test_idx = train_test_split(range(len(subjects)*8640),test_size=.1,random_state=0,shuffle=True)
trainloader = DataLoader(dataset=SSDataset(Xs,ys,train_idx),batch_size=32,shuffle=True)
devloader = DataLoader(dataset=SSDataset(Xs,ys,test_idx),batch_size=32,shuffle=True)
print(f'trainloader: {len(trainloader)} batches')
print(f'devloader: {len(devloader)} batches')

params = sum([p.flatten().size()[0] for p in list(model.parameters())])
print("Params: ",params)

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
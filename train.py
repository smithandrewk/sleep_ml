"""
author: Andrew Smith
date: Mar 21
description:
"""
import json
import argparse
from datetime import datetime
import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from lib.utils import *
from lib.models import *
from lib.datasets import Dataset2p0

# argparse
parser = argparse.ArgumentParser(description='Training program')
parser.add_argument('-r','--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument("-e", "--epochs", type=int, default=100,help="Number of training iterations")
parser.add_argument("-d", "--device", type=int, default=0,help="Cuda device to select")
parser.add_argument("-p", "--project", type=str, default='project',help="Project directory name")
parser.add_argument("-f", "--fold", type=int, default=0,help="Fold")
args = parser.parse_args()

data_dir = f'w1_cv_{args.fold}'
make_cv_data_from_ekyn(args.fold,1)
current_date = str(datetime.now()).replace(' ','_')
project_dir = args.project
early_stopping = True
patience = 50
lr = 3e-4
batch_size = 32

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
config = {
    'BATCH_SIZE':batch_size,
    'EPOCHS':args.epochs,
    'RESUME':args.resume,
    'START_TIME':current_date,
    'LEARNING_RATE':lr,
    'DATA_DIR':data_dir,
    'PATIENCE':patience
}

if not os.path.isdir(project_dir):
    os.system(f'mkdir {project_dir}')
if not os.path.isdir(f'{project_dir}/{current_date}'):
    os.system(f'mkdir {project_dir}/{current_date}')

trainloader = DataLoader(Dataset2p0(dir=f'{data_dir}/train/',labels=f'{data_dir}/y_train.pt'),batch_size=batch_size,shuffle=True)
devloader = DataLoader(Dataset2p0(dir=f'{data_dir}/dev/',labels=f'{data_dir}/y_dev.pt'),batch_size=batch_size,shuffle=True)
testloader = DataLoader(Dataset2p0(dir=f'{data_dir}/test/',labels=f'{data_dir}/y_test.pt'),batch_size=batch_size,shuffle=True)

print(f'trainloader: {len(trainloader)} batches')
print(f'devloader: {len(devloader)} batches')
print(f'testloader: {len(testloader)} batches')

model = MLP()
params = sum([p.flatten().size()[0] for p in list(model.parameters())])
print("Params: ",params)

if(config['RESUME']):
    print("Resuming previous training")
    if os.path.exists(f'{project_dir}/model.pt'):
        model.load_state_dict(torch.load(f=f'{project_dir}/model.pt'))
    else:
        print("Model file does not exist.")
        print("Exiting because resume flag was given and model does not exist. Either remove resume flag or move model to directory.")
        exit(0)
    with open(f'{project_dir}/config.json','r') as f:
        previous_config = json.load(f)
    config['START_EPOCH'] = previous_config['END_EPOCH'] + 1
    config['best_dev_loss'] = previous_config['best_dev_loss']
else:
    config['START_EPOCH'] = 0

config['END_EPOCH'] = config['START_EPOCH'] + config['EPOCHS'] - 1

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

loss_tr = []
loss_dev = []
patiencei = 0
best_model_index = 0
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

    with torch.no_grad():
        loss_dev_total = 0
        for (X_dv,y_dv) in devloader:
            X_dv,y_dv = X_dv.to(device),y_dv.to(device)
            logits = model(X_dv)
            loss = criterion(logits,y_dv)
            loss_dev_total += loss.item()
        loss_dev.append(loss_dev_total/len(devloader))
    if(early_stopping):
        if(epoch == 0):
            # first epoch
            config['best_dev_loss'] = loss_dev[-1]
        else:
            if(loss_dev[-1] < config['best_dev_loss']):
                # new best loss
                config['best_dev_loss'] = loss_dev[-1]
                patiencei = 0
                best_model_index = epoch
                torch.save(model.state_dict(), f=f'{project_dir}/best_model.pt')
                loss,metrics,y_true,y_pred,y_logits = evaluate(devloader,model,criterion,device)
                cm_grid(y_true=y_true,y_pred=y_pred,save_path=f'{project_dir}/cm_best.jpg')
            else:
                patiencei += 1
                if(patiencei == patience):
                    print("early stopping")
                    break
    best = config['best_dev_loss']
    pbar.set_description(f'\033[94m Train Loss: {loss_tr[-1]:.4f}\033[93m Dev Loss: {loss_dev[-1]:.4f}\033[92m Best Loss: {best:.4f} \033[91m Stopping: {patiencei}\033[0m')


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

loss,metrics,y_true,y_pred,y_logits = evaluate(devloader,model,criterion,device)
cm_grid(y_true=y_true,y_pred=y_pred,save_path=f'{project_dir}/{current_date}/cm_last_dev.jpg')

torch.save(model.state_dict(), f=f'{project_dir}/{current_date}/model.pt')
torch.save(model.state_dict(), f=f'{project_dir}/model.pt')

# save config
with open(f'{project_dir}/config.json', 'w') as f:
     f.write(json.dumps(config))
with open(f'{project_dir}/{current_date}/config.json', 'w') as f:
     f.write(json.dumps(config))
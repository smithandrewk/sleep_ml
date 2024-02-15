from lib.utils import *
from lib.models import *
from lib.env import *

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import os
import json
import datetime
import random
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser(description='Training program')
parser.add_argument('-r','--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument('-o','--overwrite', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument("-e", "--epochs", type=int, default=1000,help="Number of training iterations")
parser.add_argument("-d", "--device", type=int, default=0,help="Cuda device to select")
parser.add_argument("--project", type=str, default='results',help="Cuda device to select")
parser.add_argument("--window", type=int, default=5000,help="Cuda device to select")
parser.add_argument("--fold", type=int, default=0,help="Cuda device to select")
parser.add_argument("--width",nargs='+', type=int, help="Number of blocks")
parser.add_argument("--depth",nargs='+', type=int, help="Number of blocks")


args = parser.parse_args()

DATE = datetime.datetime.now().strftime("%Y-%d-%m_%H:%M")
RESUME = args.resume
OVERWRITE = args.overwrite
EPOCHS = args.epochs
DEVICE_ID = args.device
CONFIG = {
    'PATIENCE':10,
    'BEST_DEV_LOSS':torch.inf,
    'BEST_DEV_F1':0,
    'LAST_EPOCH':0,
    'TRAINLOSSI':[],
    'DEVLOSSI':[],
    'TRAINF1':[],
    'DEVF1':[],
    'BATCH_SIZE':512,
    'LEARNING_RATE':3e-4,
    'PROGRESS':0,
    'WINDOW_SIZE':args.window,
    'DEPTHI':args.depth,
    'WIDTHI':args.width
}

PROJECT_DIR = f'projects/{args.depth}_{args.width}'
writer = SummaryWriter(f'runs/{args.depth}_{args.width}')

if DEVICE == 'cuda':
    DEVICE = f'{DEVICE}:{DEVICE_ID}'

## Your Code Here (trainloader,devloader,model,criterion,optimizer) ##
from lib.datasets import EpochedDataset
from torch.utils.data import ConcatDataset
from lib.ekyn import get_ekyn_ids

idx = get_ekyn_ids()
train_idx,test_idx = idx[:8],idx[8:]

print(train_idx,test_idx)
print(len(train_idx),len(test_idx))

trainloader = DataLoader(ConcatDataset([EpochedDataset(idx=idx,condition=condition) for idx in train_idx for condition in ['Vehicle','PF']]),batch_size=512,shuffle=True)
devloader = DataLoader(ConcatDataset([EpochedDataset(idx=idx,condition=condition) for idx in test_idx for condition in ['Vehicle','PF']]),batch_size=512,shuffle=True)
model = RegNet(in_features=CONFIG['WINDOW_SIZE'],in_channels=1,depthi=CONFIG['DEPTHI'],widthi=CONFIG['WIDTHI'])
criterion = nn.CrossEntropyLoss(weight=torch.tensor([18.3846,  2.2810,  1.9716])).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(),lr=CONFIG['LEARNING_RATE'])
## End Your Code Here ##

print(model)
print("Params: ", sum([p.flatten().size()[0] for p in list(model.parameters())]))
CONFIG['MODEL'] = str(model)
model.to(DEVICE)
if RESUME:
    if not os.path.exists(f'{PROJECT_DIR}/last.pt'):
        raise Exception('cannot resume, last.pt does not exist')
    model.load_state_dict(torch.load(f=f'{PROJECT_DIR}/last.pt', map_location='cpu'))
    optimizer.load_state_dict(torch.load(f=f'{PROJECT_DIR}/last.adam.pt',map_location='cpu'))
    with open(f'{PROJECT_DIR}/config.json','r') as f:
        CONFIG = json.load(f)
    CONFIG['PROGRESS'] = 0
else:
    if not OVERWRITE and os.path.exists(PROJECT_DIR):
        print('project exists, resume or overwrite')
        exit(0)
    os.system(f'rm -rf {PROJECT_DIR}')
    os.makedirs(PROJECT_DIR)

os.makedirs(f'{PROJECT_DIR}/{DATE}')

pbar = tqdm(range(CONFIG["LAST_EPOCH"],CONFIG["LAST_EPOCH"]+EPOCHS))

for epoch in pbar:
    loss,f1 = training_loop(model=model,trainloader=trainloader,criterion=criterion,optimizer=optimizer,device=DEVICE)
    CONFIG["TRAINLOSSI"].append(loss)
    CONFIG["TRAINF1"].append(f1)
    writer.add_scalar('train loss',loss,epoch)
    writer.add_scalar('train f1',f1,epoch)

    loss,f1 = development_loop(model=model,devloader=devloader,criterion=criterion,device=DEVICE)
    CONFIG["DEVLOSSI"].append(loss)
    CONFIG["DEVF1"].append(f1)
    writer.add_scalar('dev loss',loss,epoch)
    writer.add_scalar('dev f1',f1,epoch)
    CONFIG["LAST_EPOCH"] += 1

    torch.save(model.state_dict(), f=f'{PROJECT_DIR}/last.pt')
    torch.save(model.state_dict(), f=f'{PROJECT_DIR}/{DATE}/last.pt')
    torch.save(optimizer.state_dict(), f=f'{PROJECT_DIR}/last.adam.pt')
    torch.save(optimizer.state_dict(), f=f'{PROJECT_DIR}/{DATE}/last.adam.pt')

    with open(f'{PROJECT_DIR}/config.json', 'w') as f:
        f.write(json.dumps(CONFIG))
    with open(f'{PROJECT_DIR}/{DATE}/config.json', 'w') as f:
        f.write(json.dumps(CONFIG))

    if CONFIG["DEVLOSSI"][-1] < CONFIG['BEST_DEV_LOSS']:
        CONFIG['BEST_DEV_LOSS'] = CONFIG["DEVLOSSI"][-1]
        CONFIG['BEST_DEV_LOSS_EPOCH'] = epoch
        torch.save(model.state_dict(), f=f'{PROJECT_DIR}/best.pt')
        torch.save(model.state_dict(), f=f'{PROJECT_DIR}/{DATE}/best.pt')
        torch.save(optimizer.state_dict(), f=f'{PROJECT_DIR}/best.adam.pt')
        torch.save(optimizer.state_dict(), f=f'{PROJECT_DIR}/{DATE}/best.adam.pt')
        CONFIG['PROGRESS'] = 0
    elif CONFIG['PROGRESS'] == CONFIG['PATIENCE']:
        print("early stopping")
        break
    else:
        CONFIG['PROGRESS'] += 1
    
    if CONFIG["DEVF1"][-1] > CONFIG['BEST_DEV_F1']:
        CONFIG['BEST_DEV_F1'] = CONFIG["DEVF1"][-1]
        CONFIG['BEST_DEV_F1_EPOCH'] = epoch
        torch.save(model.state_dict(), f=f'{PROJECT_DIR}/best.f1.pt')
        torch.save(model.state_dict(), f=f'{PROJECT_DIR}/{DATE}/best.f1.pt')
        torch.save(optimizer.state_dict(), f=f'{PROJECT_DIR}/best.f1.adam.pt')
        torch.save(optimizer.state_dict(), f=f'{PROJECT_DIR}/{DATE}/best.f1.adam.pt')
        CONFIG['PROGRESS'] = 0

    pbar.set_description(f'\033[94m Train Loss: {CONFIG["TRAINLOSSI"][-1]:.4f}\033[93m Dev Loss: {CONFIG["DEVLOSSI"][-1]:.4f} \033[92m Best Loss: {CONFIG["BEST_DEV_LOSS"]:.4f} \033[96m Best F1: {CONFIG["BEST_DEV_F1"]:.4f} \033[95m Progress: {CONFIG["PROGRESS"]} \033[0m')

    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
    ax[0].plot(CONFIG["TRAINLOSSI"])
    ax[0].plot(CONFIG["DEVLOSSI"])
    ax[0].axhline(y=CONFIG["BEST_DEV_LOSS"],color='r',linewidth=.5)
    ax[0].axvline(x=CONFIG["BEST_DEV_LOSS_EPOCH"],color='black',linewidth=.5)
    ax[1].plot(CONFIG["TRAINF1"])
    ax[1].plot(CONFIG["DEVF1"])
    ax[1].axhline(y=CONFIG["BEST_DEV_F1"],color='r',linewidth=.5)
    ax[1].axvline(x=CONFIG["BEST_DEV_F1_EPOCH"],color='black',linewidth=.5)
    plt.savefig(f'{PROJECT_DIR}/loss.jpg')
    plt.savefig(f'{PROJECT_DIR}/{DATE}/loss.jpg')
    plt.close()
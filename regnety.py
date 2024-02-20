from lib.utils import *
from lib.models import *

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import os
import json
import datetime
from pandas import DataFrame
import random
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Training program')
parser.add_argument('-r','--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument('--random', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument('-o','--overwrite', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument("--epochs", type=int, default=2000,help="Number of training iterations")
parser.add_argument("--device", type=int, default=0,help="Cuda device to select")
parser.add_argument("--width",nargs='+', type=int, help="Number of blocks")
parser.add_argument("--depth",nargs='+', type=int, help="Depth of each stage")
parser.add_argument("--testsize", type=float, default=.25, help="Depth of each stage")
parser.add_argument("--batch", type=int, default=256, help="Depth of each stage")
parser.add_argument("--lr", type=float, default=3e-3, help="Depth of each stage")
args = parser.parse_args()

DATE = datetime.datetime.now().strftime("%Y-%d-%m_%H:%M")
RESUME = args.resume
OVERWRITE = args.overwrite
EPOCHS = args.epochs
DEVICE_ID = args.device

CONFIG = {
    'DEVICE':'cuda',
    'PATIENCE':500,
    'PROGRESS':0,
    'DEPTHI':args.depth,
    'WIDTHI':args.width,
    'BEST_DEV_LOSS':torch.inf,
    'BEST_DEV_F1':0,
    'LAST_EPOCH':0,
    'TRAINLOSSI':[],
    'DEVLOSSI':[],
    'TRAINF1':[],
    'DEVF1':[],
    'BATCH_SIZE':args.batch,
    'LEARNING_RATE':args.lr,
    'TEST_SIZE':args.testsize,
    'STEM_KERNEL_SIZE':3,
    'WEIGHTED_LOSS':False
}
if CONFIG['DEVICE'] == 'cuda':
    DEVICE = f'{CONFIG["DEVICE"]}:{DEVICE_ID}'

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

## Your Code Here (trainloader,devloader,model,criterion,optimizer) ##
from torch.utils.data import DataLoader,ConcatDataset
from lib.ekyn import *
from lib.datasets import EpochedDataset

train_idx,test_idx = train_test_split(get_ekyn_ids(),test_size=CONFIG['TEST_SIZE'],random_state=0)
print(train_idx,test_idx)
trainloader = DataLoader(ConcatDataset([EpochedDataset(idx=idx,condition=condition) for idx in train_idx for condition in ['Vehicle','PF']]),batch_size=CONFIG['BATCH_SIZE'],shuffle=True)
devloader = DataLoader(ConcatDataset([EpochedDataset(idx=idx,condition=condition) for idx in test_idx for condition in ['Vehicle','PF']]),batch_size=CONFIG['BATCH_SIZE'],shuffle=True)
from lib.models import RegNetY
model = RegNetY(depth=CONFIG['DEPTHI'],width=CONFIG['WIDTHI'],stem_kernel_size=CONFIG['STEM_KERNEL_SIZE'])
if CONFIG['WEIGHTED_LOSS']:
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([18.3846,  2.2810,  1.9716]))
else:
    criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=CONFIG['LEARNING_RATE'])
## End Your Code Here ##

CONFIG['PARAMS'] = sum([p.flatten().size()[0] for p in list(model.parameters())])
CONFIG['MODEL'] = str(model)
model.to(DEVICE)
criterion.to(DEVICE)

## Resume functionality
PROJECTS_BASE = f'./projects'
if RESUME:
    # get last project
    if not os.path.exists(PROJECTS_BASE):
        raise FileNotFoundError(PROJECTS_BASE)
    PROJECT_NAME = int(sorted(os.listdir(f'./projects'))[-1])
    PROJECT_DIR = f'projects/{PROJECT_NAME}'

    if not os.path.exists(f'{PROJECT_DIR}/last.pt'):
        raise FileNotFoundError(f'{PROJECT_DIR}/last.pt')
    model.load_state_dict(torch.load(f=f'{PROJECT_DIR}/last.pt', map_location='cpu'))
    optimizer.load_state_dict(torch.load(f=f'{PROJECT_DIR}/last.adam.pt',map_location='cpu'))
    with open(f'{PROJECT_DIR}/config.json','r') as f:
        CONFIG = json.load(f)
    CONFIG['PROGRESS'] = 0
else:
    if not os.path.exists(PROJECTS_BASE):
        print('projects directory dne, making')
        os.makedirs(PROJECTS_BASE)
        PROJECT_NAME = '0' # first project is always 0
    else:
        project_list_int = sorted([int(file) for file in os.listdir(PROJECTS_BASE)])
        print(f'projects directory exists, finding last project name')
        PROJECT_NAME = project_list_int[-1] + 1 # add 1 to the last project name
    print(f'making project {PROJECT_NAME}')
    PROJECT_DIR = f'projects/{PROJECT_NAME}'
    os.makedirs(PROJECT_DIR)
    writer = SummaryWriter(f'runs/{PROJECT_NAME}')

os.makedirs(f'{PROJECT_DIR}/{DATE}')

print(DataFrame([CONFIG])[['DEPTHI','WIDTHI','PARAMS']])
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
    ax[0].plot(torch.linspace(0,len(CONFIG["TRAINLOSSI"]),len(CONFIG["DEVLOSSI"])),CONFIG["DEVLOSSI"])
    ax[0].axhline(y=CONFIG["BEST_DEV_LOSS"],color='r',linewidth=.5)
    ax[0].axvline(x=CONFIG["BEST_DEV_LOSS_EPOCH"],color='black',linewidth=.5)
    ax[1].plot(CONFIG["TRAINF1"])
    ax[1].plot(torch.linspace(0,len(CONFIG["TRAINF1"]),len(CONFIG["DEVF1"])),CONFIG["DEVF1"])
    ax[1].axhline(y=CONFIG["BEST_DEV_F1"],color='r',linewidth=.5)
    ax[1].axvline(x=CONFIG["BEST_DEV_F1_EPOCH"],color='black',linewidth=.5)
    plt.savefig(f'{PROJECT_DIR}/loss.jpg')
    plt.savefig(f'{PROJECT_DIR}/{DATE}/loss.jpg')
    plt.close()
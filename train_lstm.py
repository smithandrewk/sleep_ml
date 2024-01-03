from lib.utils import *
from lib.models import *
from lib.env import *

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
from torch import cat,zeros
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import os
import json
import datetime

parser = argparse.ArgumentParser(description='Training program')
parser.add_argument('-r','--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument('-o','--overwrite', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument("-e", "--epochs", type=int, default=100,help="Number of training iterations")
parser.add_argument("-d", "--device", type=int, default=0,help="Cuda device to select")
parser.add_argument("-p", "--project", type=str, default='results',help="Cuda device to select")
parser.add_argument("-w", "--window", type=int, default='1',help="Cuda device to select")
parser.add_argument("-b", "--blocks", type=int, default='2',help="Cuda device to select")
args = parser.parse_args()

BLOCKS = (3,4,8,16,32,64)
DATE = datetime.datetime.now().strftime("%Y-%d-%m_%H:%M")
RESUME = args.resume
OVERWRITE = args.overwrite
EPOCHS = args.epochs
DEVICE_ID = args.device
PATIENCE = 20
CONFIG = {
    'BEST_DEV_LOSS':torch.inf,
    'BEST_DEV_F1':0,
    'LAST_EPOCH':0,
    'TRAINLOSSI':[],
    'DEVLOSSI':[],
    'TRAINF1':[],
    'DEVF1':[],
    'WINDOWSIZE':args.window,
    'BATCH_SIZE':512,
    'LEARNING_RATE':3e-4,
    'BLOCK_SIZES':BLOCKS[:args.blocks],
    'PATIENCE':PATIENCE,
    'PROGRESS':0
}

PROJECT_DIR = f'../projects/{args.project}'

if DEVICE == 'cuda':
    DEVICE = f'{DEVICE}:{DEVICE_ID}'

class Windowset(Dataset):
    def __init__(self,X,y,windowsize):
        self.windowsize = windowsize
        self.X = cat([zeros(windowsize // 2,5000),X,zeros(windowsize // 2,5000)])
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx:idx+self.windowsize],self.y[idx])
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
from lib.ekyn import *
idx = get_ekyn_ids()
train_idx,test_idx = train_test_split(idx,test_size=.25,random_state=0)
print(train_idx,test_idx)
print(len(train_idx),len(test_idx))
train_idx = train_idx
test_idx = test_idx
print(train_idx,test_idx)
X,y = load_eeg_label_pairs(ids=train_idx)
trainloader = DataLoader(Windowset(X,y,CONFIG['WINDOWSIZE']),batch_size=512,shuffle=True)
X,y = load_eeg_label_pairs(ids=test_idx)
devloader = DataLoader(Windowset(X,y,CONFIG['WINDOWSIZE']),batch_size=512,shuffle=True)
from lib import models

class MyLSTM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = models.ResNetv3(windowsize=1,starting_filters=2,n_blocks=2)
        self.lstm = nn.LSTM(input_size=4,hidden_size=32,batch_first=True)
        self.classifier = nn.Linear(in_features=32,out_features=3)
    def forward(self,x):
        x = self.encoder(x).reshape(-1,3,4)
        _,(h,_) = self.lstm(x)
        x = self.classifier(h.squeeze())
        return x
model = MyLSTM()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=CONFIG['LEARNING_RATE'])
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
        raise Exception('project exists, either resume or pass -o to overwrite')
    os.system(f'rm -rf {PROJECT_DIR}')
    os.makedirs(PROJECT_DIR)

os.makedirs(f'{PROJECT_DIR}/{DATE}')

pbar = tqdm(range(CONFIG["LAST_EPOCH"],CONFIG["LAST_EPOCH"]+EPOCHS))

for epoch in pbar:
    loss,f1 = training_loop(model=model,trainloader=trainloader,criterion=criterion,optimizer=optimizer,device=DEVICE)
    CONFIG["TRAINLOSSI"].append(loss)
    CONFIG["TRAINF1"].append(f1)
    loss,f1 = development_loop(model=model,devloader=devloader,criterion=criterion,device=DEVICE)
    CONFIG["DEVLOSSI"].append(loss)
    CONFIG["DEVF1"].append(f1)
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
    elif CONFIG['PROGRESS'] == PATIENCE:
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
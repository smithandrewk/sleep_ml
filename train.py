from lib.ekyn import *
import matplotlib.pyplot as plt
from lib.models import MLP
from torch.utils.data import DataLoader,ConcatDataset
from lib.deep_learning_utils import count_params
import torch
import os
from lib.env import DATA_PATH,CONDITIONS
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from lib.deep_learning_utils import training_loop,development_loop

if not os.path.isdir(f'{DATA_PATH}/projects'):
    os.makedirs(f'{DATA_PATH}/projects')

existing_projects = sorted([int(dir) for dir in os.listdir(f'{DATA_PATH}/projects')])
if len(existing_projects) == 0:
    PROJECT_ID = 0
else:
    PROJECT_ID = existing_projects[-1] + 1

PROJECT_PATH = f'{DATA_PATH}/projects/{PROJECT_ID}'
if not os.path.isdir(PROJECT_PATH):
    os.makedirs(PROJECT_PATH)

EPOCHS = 1000
CONFIG = {
    'BATCH_SIZE':2048,
    'TRAIN_IDS':[],
    'TEST_IDS':[],
    'TRAINLOSS':[],
    'TRAINF1':[],
    'DEVLOSS':[],
    'DEVF1':[],
    'BEST_DEV_LOSS':torch.inf,
    'BEST_DEV_F1':0,
    'BEST_DEV_LOSS_EPOCH':0,
    'BEST_DEV_F1_EPOCH':0,  
    'LAST_EPOCH':0,
    'PROGRESS':0,
    'PATIENCE':100,
    'DEVICE':'cuda'
}

# %%
CONFIG['TRAIN_IDS'],CONFIG['TEST_IDS'] = train_test_split(get_ekyn_ids(),test_size=.25,random_state=0)
# %%
from lib.models import ResidualBlock
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,in_feature_maps,out_feature_maps,n_features) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(in_feature_maps,out_feature_maps,kernel_size=8,padding='same',bias=False)
        self.bn1 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

        self.c2 = nn.Conv1d(out_feature_maps,out_feature_maps,kernel_size=5,padding='same',bias=False)
        self.bn2 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

        self.c3 = nn.Conv1d(out_feature_maps,out_feature_maps,kernel_size=3,padding='same',bias=False)
        self.bn3 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

        self.c4 = nn.Conv1d(in_feature_maps,out_feature_maps,1,padding='same',bias=False)
        self.bn4 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

    def forward(self,x):
        identity = x
        x = self.c1(x)
        x = self.bn1(x)
        x = relu(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = relu(x)

        x = self.c3(x)
        x = self.bn3(x)
        x = relu(x)

        identity = self.c4(identity)
        identity = self.bn4(identity)

        x = x+identity
        x = relu(x)
        
        return x

class FrodoSmall(nn.Module):
    """
    the little wanderer
    """
    def __init__(self,n_features) -> None:
        super().__init__()
        self.n_features = n_features
        self.block1 = ResidualBlock(1,64,n_features)

        self.gap = nn.AvgPool1d(kernel_size=n_features)
        self.fc1 = nn.Linear(in_features=64,out_features=3)
    def forward(self,x,classification=True):
        x = x.view(-1,1,self.n_features)
        x = self.block1(x)
        x = self.gap(x)
        if(classification):
            x = self.fc1(x.squeeze())
            return x
        else:
            return x.squeeze()
    def get_dataloaders(self,batch_size,train_ids,test_ids):
        trainloader = DataLoader(
            dataset=ConcatDataset(
            [EpochedDataset(id=id,condition=condition) for id in train_ids for condition in CONDITIONS]
            ),
            batch_size=batch_size,
            shuffle=True
        )
        testloader = DataLoader(
            dataset=ConcatDataset(
            [EpochedDataset(id=id,condition=condition) for id in test_ids for condition in CONDITIONS]
            ),
            batch_size=batch_size,
            shuffle=False
        )
        print(len(trainloader)*batch_size)
        print(len(testloader)*batch_size)
        return trainloader,testloader
    

model = FrodoSmall(n_features=1000)
model.to(CONFIG['DEVICE'])
trainloader,testloader = model.get_dataloaders(CONFIG['BATCH_SIZE'],CONFIG['TRAIN_IDS'],CONFIG['TEST_IDS'])
# model = torch.compile(model)
# torch.set_float32_matmul_precision('high')
print(f'{count_params(model)} parameters')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# %%
model.train()

pbar = tqdm(range(CONFIG["LAST_EPOCH"],CONFIG["LAST_EPOCH"]+EPOCHS))

import json
for epoch in pbar:
    loss,f1 = training_loop(model=model,trainloader=trainloader,criterion=criterion,optimizer=optimizer,device=CONFIG['DEVICE'])
    CONFIG['TRAINLOSS'].append(loss)
    CONFIG['TRAINF1'].append(f1)

    loss,f1 = development_loop(model=model,devloader=testloader,criterion=criterion,device=CONFIG['DEVICE'])
    CONFIG['DEVLOSS'].append(loss)
    CONFIG['DEVF1'].append(f1)

    scheduler.step(loss)

    CONFIG["LAST_EPOCH"] += 1

    torch.save(model.state_dict(), f=f'{PROJECT_PATH}/last.pt')
    torch.save(optimizer.state_dict(), f=f'{PROJECT_PATH}/last.adam.pt')

    with open(f'{PROJECT_PATH}/config.json', 'w') as f:
        f.write(json.dumps(CONFIG))

    if CONFIG["DEVLOSS"][-1] < CONFIG['BEST_DEV_LOSS']: # new best dev loss
        CONFIG['BEST_DEV_LOSS'] = CONFIG["DEVLOSS"][-1]
        CONFIG['BEST_DEV_LOSS_EPOCH'] = epoch
        torch.save(model.state_dict(), f=f'{PROJECT_PATH}/best.devloss.pt')
        torch.save(optimizer.state_dict(), f=f'{PROJECT_PATH}/best.adam.devloss.pt')
        CONFIG['PROGRESS'] = 0
    if CONFIG["DEVF1"][-1] > CONFIG['BEST_DEV_F1']: # new best dev F1
        CONFIG['BEST_DEV_F1'] = CONFIG["DEVF1"][-1]
        CONFIG['BEST_DEV_F1_EPOCH'] = epoch
        torch.save(model.state_dict(), f=f'{PROJECT_PATH}/best.devF1.pt')
        torch.save(optimizer.state_dict(), f=f'{PROJECT_PATH}/best.adam.devF1.pt')
        CONFIG['PROGRESS'] = 0
    if CONFIG['PROGRESS'] == CONFIG['PATIENCE']:
        print('early stopping')
        break

    CONFIG['PROGRESS'] += 1

    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(13,4),dpi=200)
    ax[0].plot(CONFIG['TRAINLOSS'])
    ax[0].plot(CONFIG['DEVLOSS'])
    ax[0].set_title(label='loss',fontweight='bold')
    ax[0].set_xlabel('Epoch')
    ax[1].plot(CONFIG['TRAINF1'])
    ax[1].plot(CONFIG['DEVF1'])
    ax[1].set_title(label='f1',fontweight='bold')
    ax[1].set_xlabel('Epoch')

    plt.savefig('loss.jpg')
    plt.savefig(f'{PROJECT_PATH}/loss.jpg')
    plt.close()

    pbar.set_description(f'\033[94m Train Loss: {CONFIG["TRAINLOSS"][-1]:.4f}\033[93m Dev Loss: {CONFIG["DEVLOSS"][-1]:.4f} \033[92m Best Loss: {CONFIG["BEST_DEV_LOSS"]:.4f} \033[96m Best F1: {CONFIG["BEST_DEV_F1"]:.4f} \033[95m Progress: {CONFIG["PROGRESS"]} \033[0m')

# %%
torch.save(model.state_dict(),f'{PROJECT_PATH}/model.last.pt')

# %%
from lib.deep_learning_utils import evaluate
from sklearn.metrics import ConfusionMatrixDisplay,classification_report

loss,report,y_true,y_pred,y_logits = evaluate(dataloader=testloader,model=model,criterion=criterion)
ConfusionMatrixDisplay.from_predictions(y_true,y_pred,normalize='true')
plt.savefig(f'{PROJECT_PATH}/cm.jpg')
print(classification_report(y_true,y_pred))
print(loss)



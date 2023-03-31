"""
author: Andrew Smith
date: Mar 21
description:
"""
import argparse
import os
from datetime import datetime
import json

from lib.utils import *
from lib.models import SimpleCNN as MODEL
from lib.datasets import EEGDataset
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

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

train_dataloader = DataLoader(EEGDataset(dir='data/pt_bal/train',labels='data/pt_bal/y_train.pt'), batch_size=args.batch, shuffle=True)
test_dataloader = DataLoader(EEGDataset(dir='data/pt_bal/test',labels='data/pt_bal/y_test.pt'), batch_size=args.batch, shuffle=False)

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
optimizer = torch.optim.Adam(model.parameters())
training_losses = []
testing_losses = []

pbar = tqdm(range(config['EPOCHS']))
for epoch in pbar:
    training_loss = 0
    for (X,y) in tqdm(train_dataloader):
        X,y = X.to(device),y.to(device)
        logits = model(X)
        loss = criterion(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    training_loss = training_loss/len(train_dataloader)
    training_losses.append(training_loss)
    model.eval()
    testing_loss = 0
    for (X,y) in test_dataloader:
        X,y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits,y)
        testing_loss += loss.item()
    testing_loss = testing_loss/len(test_dataloader)
    testing_losses.append(testing_loss)
    pbar.set_description(f'\033[94mDev Loss: {training_loss:.4f}\033[93m Val Loss: {testing_loss:.4f}\033[0m')
    plt.plot(training_losses)
    plt.plot(testing_losses)
    plt.savefig(f'project/{current_date}/loss.jpg')
    plt.close()

# test confusion matrices
y_true = torch.Tensor()
y_pred = torch.Tensor().cuda()
for (X,y) in test_dataloader:
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
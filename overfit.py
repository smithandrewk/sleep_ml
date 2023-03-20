
print("Running main.py")

## Imports
from lib.utils import *
from lib.models import MLP
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

##Argparse
parser = argparse.ArgumentParser(description='Training program')
parser.add_argument('-r','--resume', help='Whether to resume previous training',required=False,action='store_true',default=False)
parser.add_argument("-e", "--epochs", type=int, default=100,help="Number of training iterations")
parser.add_argument("-d", "--device", type=int, default=0,help="Cuda device to select")
args = parser.parse_args()


## Load Data
X,y = load_raw_list([21])
y = one_hot(y,num_classes=3).reshape(-1,3).float()
X = X.flatten()[::10]
X = X.reshape(-1,500)


device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(TensorDataset(X,y), batch_size=256, shuffle=True)

model = MLP()

if(args.resume):
    print("Resuming previous training")
    if os.path.exists(f'./model.pt'):
        model.load_state_dict(torch.load(f='model.pt'))
    else:
        print("Model file does not exist.")
        print("Exiting because resume flag was given and model does not exist. Either remove resume flag or move model to directory.")
        exit(0)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
training_losses = []


# Get initial loss
total_loss = 0
for (X,y) in dataloader:
    X,y = X.to(device),y.to(device)
    logits = model(X)
    loss = criterion(logits,y)
    total_loss += loss.item()
print("initial loss",total_loss/len(dataloader))

pbar = tqdm(range(args.epochs))
for epoch in pbar:
    training_loss = 0
    for (X,y) in dataloader:
        X,y = X.to(device),y.to(device)
        logits = model(X)
        loss = criterion(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    training_loss = training_loss/len(dataloader)
    training_losses.append(training_loss)
    pbar.set_description(f'\033[93mDev Loss: {training_loss:.4f}\033[0m')


plt.plot(training_losses)
plt.savefig('loss.jpg')

from datetime import datetime
current_date = str(datetime.now()).replace(' ','_')
if not os.path.isdir('models'):
    os.system('mkdir models')

if torch.cuda.device_count() > 1:
    torch.save(model.module.state_dict(), f=f'models/{current_date}.pt')
    torch.save(model.module.state_dict(), f=f'model.pt')
else:
    torch.save(model.state_dict(), f=f'models/{current_date}.pt')
    torch.save(model.state_dict(), f=f'model.pt')

print("Running main.py")

## Imports
from lib.utils import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

##Argparse
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-r','--resume', help='Description for foo argument',required=False,action='store_true',default=False)
parser.add_argument("-e", "--epochs", type=int, default=100,
                    help="The number of games to simulate")
parser.add_argument("-d", "--device", type=int, default=0,options=[0,1],help="The number of games to simulate")
args = parser.parse_args()

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

## Load Data
X,y = load_raw_list([21])
y = one_hot(y,num_classes=3).reshape(-1,3).float()
X = X.flatten()[::10]
X = X.reshape(-1,500)

dataloader = DataLoader(TensorDataset(X,y), batch_size=1024, shuffle=True)

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=500,out_features=32)
        self.fc2 = nn.Linear(in_features=32,out_features=3)

    def forward(self,x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        return x

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
    model.train()
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
    # if(epoch % 10 == 0):
        # print(training_loss/len(dataloader))
    # if(epoch%50 == 0):
        # plt.plot(training_losses)
        # plt.savefig('loss.jpg')
        # plt.close()
plt.plot(training_losses)
plt.savefig('loss.jpg')
torch.save(model.module.state_dict(), f='model.pt')
print("Running main.py")

## Imports
from lib.utils import *
from lib.models import CNN_0
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from torch.nn.functional import softmax

##Argparse
parser = argparse.ArgumentParser(description='Training program')
parser.add_argument('-r','--resume', help='Whether to resume previous training',required=False,action='store_true',default=False)
parser.add_argument("-e", "--epochs", type=int, default=100,help="Number of training iterations")
parser.add_argument("-d", "--device", type=int, default=0,help="Cuda device to select")
parser.add_argument("-b", "--batch", type=int, default=64,help="Batch Size")

args = parser.parse_args()

BATCH_SIZE = args.batch
EPOCHS = args.epochs
RESUME = args.resume

## Model Saving
from datetime import datetime
current_date = str(datetime.now()).replace(' ','_')
if not os.path.isdir('models'):
    os.system('mkdir models')
os.system(f'mkdir models/{current_date}')

## Load Data
X,y = load_raw_list([20,21,22,23,24,25])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,shuffle=True,stratify=y,random_state=0)
X_train = X_train.reshape(-1,1,5000)
X_test = X_test.reshape(-1,1,5000)
y_train_one_hot = one_hot(y_train,num_classes=3).reshape(-1,3).float()
y_test_one_hot = one_hot(y_test,num_classes=3).reshape(-1,3).float()
train_dataloader = DataLoader(TensorDataset(X_train,y_train_one_hot), batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(TensorDataset(X_test,y_test_one_hot), batch_size=BATCH_SIZE, shuffle=False)


## Model
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

model = CNN_0()

if(RESUME):
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
testing_losses = []

# Get initial loss
total_loss = 0
for (X,y) in train_dataloader:
    X,y = X.to(device),y.to(device)
    logits = model(X)
    loss = criterion(logits,y)
    total_loss += loss.item()
print("initial loss",total_loss/len(train_dataloader))

pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    training_loss = 0
    for (X,y) in train_dataloader:
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

    plt.plot(training_losses[-400:])
    plt.plot(testing_losses[-400:])
    plt.savefig('loss.jpg')
    plt.close()

plt.plot(training_losses)
plt.plot(testing_losses)
plt.savefig('loss.jpg')
plt.close()

y_pred = torch.Tensor().cuda()
for (X,_) in test_dataloader:
    y_pred = torch.cat([y_pred,softmax(model(X.cuda()),dim=1).argmax(axis=1)])
y_pred = y_pred.cpu()

cms(y_true=y_test,y_pred=y_pred)


    
if torch.cuda.device_count() > 1:
    torch.save(model.module.state_dict(), f=f'models/{current_date}.pt')
    torch.save(model.module.state_dict(), f=f'model.pt')
else:
    torch.save(model.state_dict(), f=f'models/{current_date}.pt')
    torch.save(model.state_dict(), f=f'model.pt')

from lib.utils import *
from lib.ekyn import *
import json
from tqdm import tqdm

train_size = .95
window_size = 1
data_dir = f'w{window_size}_ss'

normalize = False
ids = get_ekyn_ids()

os.makedirs(data_dir)
os.makedirs(f'{data_dir}/train')
os.makedirs(f'{data_dir}/dev')
os.makedirs(f'{data_dir}/test')

config = {
    'TRAIN_SIZE':train_size,
    'NORMALIZED':normalize,
}

with open(f'{data_dir}/config.json', 'w') as f:
     f.write(json.dumps(config))

y_train_all = torch.Tensor()
y_dev_all = torch.Tensor()
y_test_all = torch.Tensor()
x_train_i = 0
x_dev_i = 0
x_test_i = 0

for id in tqdm(ids):
    for condition in ['PF','Vehicle']:
        X,y = load_eeg_label_pair(id=id,condition=condition)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,shuffle=True,random_state=0)
        X_train,X_dev,y_train,y_dev = train_test_split(X_train,y_train,test_size=.25,shuffle=True,random_state=0)
        for Xi in X_train:
            torch.save(Xi.clone(),f'{data_dir}/train/{x_train_i}.pt')
            x_train_i += 1
        for Xi in X_dev:
            torch.save(Xi.clone(),f'{data_dir}/dev/{x_dev_i}.pt')
            x_dev_i += 1
        for Xi in X_test:
            torch.save(Xi.clone(),f'{data_dir}/test/{x_test_i}.pt')
            x_test_i += 1
        y_train_all = torch.cat([y_train_all,y_train])
        y_dev_all = torch.cat([y_dev_all,y_dev])
        y_test_all = torch.cat([y_test_all,y_test])

torch.save(y_train_all,f'{data_dir}/y_train.pt')
torch.save(y_dev_all,f'{data_dir}/y_dev.pt')
torch.save(y_test_all,f'{data_dir}/y_test.pt')

from lib.datasets import Dataset2p0
from torch.utils.data import DataLoader
trainloader = DataLoader(Dataset2p0(dir=f'{data_dir}/train/',labels=f'{data_dir}/y_train.pt'),batch_size=1,shuffle=True)
devloader = DataLoader(Dataset2p0(dir=f'{data_dir}/dev/',labels=f'{data_dir}/y_dev.pt'),batch_size=1,shuffle=True)
testloader = DataLoader(Dataset2p0(dir=f'{data_dir}/test/',labels=f'{data_dir}/y_test.pt'),batch_size=1,shuffle=True)
X,y = next(iter(trainloader))
X,y = next(iter(devloader))
X,y = next(iter(testloader))
print(X.shape,y.shape)
print(len(trainloader)+len(devloader)+len(testloader))
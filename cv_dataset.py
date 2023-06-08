from lib.utils import *
import json
from tqdm import tqdm

# TODO for each fold
foldi = 0
train_size = .95
window_size = 1
data_dir = f'w{window_size}_cv_{foldi}'
x_train_i = 0
x_dev_i = 0
normalize = False
train_ids,test_ids = get_cross_validation_split_for_fold(foldi=foldi)

os.makedirs(data_dir)
os.makedirs(f'{data_dir}/train')
os.makedirs(f'{data_dir}/dev')

config = {
    'TRAIN_SIZE':train_size,
    'TRAIN_IDS':train_ids,
    'TEST_IDS':test_ids,
    'NORMALIZED':normalize,
}

with open(f'{data_dir}/config.json', 'w') as f:
     f.write(json.dumps(config))

y_train_all = torch.Tensor()
y_dev_all = torch.Tensor()

for id in tqdm(train_ids):
    for condition in ['PF','Vehicle']:
        X,y = load_eeg_label_pair(id=id,condition=condition)
        if(normalize):
            # center, stretch
            X = (X - X.mean(axis=1,keepdim=True))/X.std(axis=1,keepdim=True)
            # drop row if any element is inf
            not_inf_idx = torch.where(~X.isinf().any(axis=1))[0]
            X,y = X[not_inf_idx], y[not_inf_idx]
        # train test split for each file, approximates the same for train-test-splitting the entire set
        X_train,X_dev,y_train,y_dev = train_test_split(X,y,test_size=(1-train_size),shuffle=True,stratify=y,random_state=0)
        for Xi in X_train:
            torch.save(Xi.clone(),f'{data_dir}/train/{x_train_i}.pt')
            x_train_i += 1
        for Xi in X_dev:
            torch.save(Xi.clone(),f'{data_dir}/dev/{x_dev_i}.pt')
            x_dev_i += 1
        y_train_all = torch.cat([y_train_all,y_train])
        y_dev_all = torch.cat([y_dev_all,y_dev])

torch.save(y_train_all,f'{data_dir}/y_train.pt')
torch.save(y_dev_all,f'{data_dir}/y_dev.pt')

from lib.datasets import Dataset2p0
from torch.utils.data import DataLoader
trainloader = DataLoader(Dataset2p0(dir=f'{data_dir}/train/',labels=f'{data_dir}/y_train.pt'),batch_size=1,shuffle=True)
devloader = DataLoader(Dataset2p0(dir=f'{data_dir}/dev/',labels=f'{data_dir}/y_dev.pt'),batch_size=1,shuffle=True)
X,y = next(iter(trainloader))
X,y = next(iter(devloader))
print(X.shape,y.shape)
print(len(trainloader)+len(devloader))
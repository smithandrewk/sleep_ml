import os
from torch import load
from lib.env import DATA_PATH
import torch

CONDITIONS = ['Vehicle','PF']
def get_ekyn_ids():
    possible_datasets = ['pt_ekyn_robust_50hz','pt_ekyn']
    for possible_dataset in possible_datasets:
        DATASET_PATH = f'{DATA_PATH}/{possible_dataset}'
        if not os.path.exists(DATASET_PATH):
            continue
        return sorted(
            list(
                set(
                    [id.split('_')[0] for id in os.listdir(DATASET_PATH)]
                    )
                )
            )
    return None

def load_ekyn_pt(id,condition):
    return load(f'{DATA_PATH}/pt_ekyn/{id}_{condition}.pt')

def load_ekyn_pt_robust(id,condition,downsampled):
    if downsampled:
        return load(f'{DATA_PATH}/pt_ekyn_robust_50hz/{id}_{condition}.pt')
    else:
        return load(f'{DATA_PATH}/pt_ekyn_robust/{id}_{condition}.pt')

def get_snezana_mice_ids():
    DATASET_PATH = f'{DATA_PATH}/pt_snezana_mice_robust_50hz'
    return [filename.replace('.pt','') for filename in os.listdir(DATASET_PATH)]

def load_snezana_mice(id):
    return load(f'{DATA_PATH}/pt_snezana_mice_robust_50hz/{id}.pt')

class EpochedDataset(torch.utils.data.Dataset):
    def __init__(self,id='A1-1',condition='Vehicle',robust=True,downsampled=True,convolution=True,snezana_mice=False):
        if snezana_mice:
            X,y = load_snezana_mice(id)
        else:
            if robust:
                X,y = load_ekyn_pt_robust(id=id,condition=condition,downsampled=downsampled)
            else:
                X,y = load_ekyn_pt(id=id,condition=condition)

        self.X = X
        self.y = y
        self.id = id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx:idx+1],self.y[idx])
    
def get_dataloaders(batch_size=512,shuffle_train=True,shuffle_test=False):
    from sklearn.model_selection import train_test_split
    ekyn_ids = get_ekyn_ids()

    train_ids,test_ids = train_test_split(ekyn_ids,test_size=.2,shuffle=True,random_state=0)

    from torch.utils.data import DataLoader,ConcatDataset
    trainloader = DataLoader(
            dataset=ConcatDataset(
            [EpochedDataset(id=id,condition=condition,robust=True,downsampled=True) for id in train_ids for condition in CONDITIONS] 
            ),
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=1
        )
    testloader = DataLoader(
            dataset=ConcatDataset(
            [EpochedDataset(id=id,condition=condition,robust=True,downsampled=True) for id in test_ids for condition in CONDITIONS] 
            ),
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=1
        )
    print('train_ids',train_ids)
    print('test_ids',test_ids)
    print('n ids',len(ekyn_ids))
    print(f'{len(trainloader)} training batches {len(testloader)} testing batches')
    print(f'{len(trainloader)*batch_size} training samples {len(testloader)*batch_size} testing samples')
    print(f'{len(trainloader)*batch_size*10/3600:.2f} training hours {len(testloader)*batch_size*10/3600:.2f} testing hours')
    return trainloader,testloader
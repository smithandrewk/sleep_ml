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
    
class EpochedDataset(torch.utils.data.Dataset):
    def __init__(self,id='A1-1',condition='Vehicle',robust=True,downsampled=True,convolution=True):
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
    
from torch.utils.data import DataLoader,ConcatDataset
def get_ratloader(ids=['A1-1']):
    return DataLoader(
        dataset=ConcatDataset(
        [EpochedDataset(id=id,condition=condition,robust=True,downsampled=True) for id in ids for condition in CONDITIONS]
        ),
        batch_size=1024,
        shuffle=True,
        num_workers=0
    )
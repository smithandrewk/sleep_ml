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
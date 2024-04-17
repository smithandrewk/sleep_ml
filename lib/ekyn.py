from os import listdir
from torch import load
from lib.env import DATA_PATH

def get_ekyn_ids():
    return sorted(list(set([id.split('_')[0] for id in listdir(f'{DATA_PATH}/pt_ekyn')])))

def load_ekyn_pt(id,condition):
    return load(f'{DATA_PATH}/pt_ekyn/{id}_{condition}.pt')

def load_ekyn_pt_robust(id,condition,downsampled):
    if downsampled:
        return load(f'{DATA_PATH}/pt_ekyn_robust_50hz/{id}_{condition}.pt')
    else:
        return load(f'{DATA_PATH}/pt_ekyn_robust/{id}_{condition}.pt')

def get_snezana_mice_ids():
    return sorted(list(set([id.split('.')[0] for id in listdir(f'{DATA_PATH}/pt_snezana_mice')])))

def load_snezana_mice_pt(id):
    return load(f'{DATA_PATH}/pt_snezana_mice/{id}.pt')

def load_snezana_mice_pt_robust(id,downsampled):
    if downsampled:
        return load(f'{DATA_PATH}/pt_snezana_mice_robust_50hz/{id}.pt')
    else:
        return load(f'{DATA_PATH}/pt_snezana_mice_robust/{id}.pt')
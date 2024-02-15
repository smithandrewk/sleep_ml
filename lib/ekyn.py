from os import listdir
from torch import load
from lib.env import DATA_PATH

def get_ekyn_ids():
    return sorted(list(set([id.split('_')[0] for id in listdir(f'{DATA_PATH}/pt_ekyn')])))

def load_ekyn_pt(idx,condition):
    return load(f'{DATA_PATH}/pt_ekyn/{idx}_{condition}.pt')

def load_ekyn_pt_standardized(idx,condition):
    return load(f'{DATA_PATH}/pt_ekyn_standardized/{idx}_{condition}.pt')

def load_ekyn_pt_robust_scaled(idx,condition):
    return load(f'{DATA_PATH}/robust_scaled/{idx}_{condition}.pt')
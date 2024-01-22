from os import listdir
from torch import load

DATA_PATH = f'/home/andrew/.aurora'

def get_ekyn_ids():
    return sorted(listdir(f'{DATA_PATH}/ekyn'))

def load_ekyn_pt(idx,condition):
    return load(f'/home/andrew/.aurora/pt_ekyn/{idx}_{condition}.pt')

def load_ekyn_pt_standardized(idx,condition):
    return load(f'/home/andrew/.aurora/pt_ekyn_standardized/{idx}_{condition}.pt')

def load_ekyn_pt_robust_scaled(idx,condition):
    return load(f'/home/andrew/.aurora/robust_scaled/{idx}_{condition}.pt')
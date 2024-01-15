from lib.env import *
from os import listdir
from torch import from_numpy, Tensor, cat, zeros
from torch.nn.functional import one_hot
from pandas import read_csv, NA, Categorical
from random import seed, shuffle
from mne.io import read_raw_edf

def load_raw_edf_by_path(path):
    raw = read_raw_edf(path,verbose=False)
    raw.rename_channels({'EEG 1':'EEG','EEG 2':'EMG'})
    raw.set_channel_types({'EEG':'eeg','EMG':'emg'},verbose=False)
    return raw

def get_ekyn_ids(DATA_PATH=DATA_PATH):
    return sorted(listdir(f'{DATA_PATH}/ekyn'))

def load_raw_edf(id='A1-1', condition='Vehicle', DATA_PATH=DATA_PATH):
    return load_raw_edf_by_path(f'{DATA_PATH}/ekyn/{id}/{condition}.edf')

def load_eeg(id='A1-1', condition='Vehicle'):
    raw = load_raw_edf(id=id, condition=condition)
    return raw.get_data(picks='EEG')[0]

def load_epoched_eeg(id='A1-1', condition='Vehicle'):
    return from_numpy(load_eeg(id=id, condition=condition).reshape(-1, 5000)).float()

def load_labels(id='A1-1', condition='Vehicle', DATA_PATH=DATA_PATH):
    df = read_csv(f'{DATA_PATH}/ekyn/{id}/{condition}.csv')
    df[df['label'] == 'X'] = NA
    df = df.ffill()
    return from_numpy(Categorical(df['label']).codes.copy()).long()

def load_one_hot_labels(id='A1-1', condition='Vehicle', DATA_PATH=DATA_PATH):
    return one_hot(load_labels(id, condition, DATA_PATH)).float()

def load_eeg_label_pair(id='A1-1', condition='Vehicle',zero_pad=False,windowsize=9):
    if(zero_pad):
        return (cat([zeros(windowsize//2,5000),load_epoched_eeg(id=id, condition=condition),zeros(windowsize//2,5000)]), load_one_hot_labels(id=id, condition=condition))
    else:
        return (load_epoched_eeg(id=id, condition=condition), load_one_hot_labels(id=id, condition=condition))

def load_eeg_label_pairs(ids):
    X = []
    y = []
    for id in ids:
        for condition in ['Vehicle', 'PF']:
            Xi,yi = load_eeg_label_pair(id,condition)
            X.append(Xi)
            y.append(yi)
    X = cat(X)
    y = cat(y)
    return X,y
from lib.env import *
from os import listdir
from lib.utils import load_raw_edf_by_path
from torch import from_numpy, Tensor, cat, zeros
from torch.nn.functional import one_hot
from pandas import read_csv, NA, Categorical
from random import seed, shuffle
from lib.utils import window_epoched_signal

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
    df = df.fillna(method='ffill')
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

def get_k_fold_cv_ids_for_ekyn(k=4):
    """
    warning: only tested for k=4. might work for things that divide the number of subjects. probably doesn't work otherwise. -andrew.
    """
    ret = []
    for foldi in range(k):
        ids = get_ekyn_ids()
        seed(0)
        shuffle(ids)
        start = foldi*k
        stop = foldi*k+int(len(ids)/k)
        test_ids = ids[start:stop]
        for id in test_ids:
            ids.remove(id)
        ret.append((ids,test_ids))
    return ret

def get_leave_one_out_cv_ids_for_ekyn():
    ids = get_ekyn_ids()
    seed(0)
    shuffle(ids)
    ret = []
    for test_id in ids:
        train_ids = [x for x in ids if x != test_id]
        ret.append((train_ids, [test_id]))
    return ret

def load_psd(id='A1-1', condition='Vehicle'):
    df = read_csv(f'{DATA_PATH}/ekyn/{id}/{condition}.csv')
    df = df.drop(['Time Stamp', 'label'], axis=1)
    return from_numpy(df.to_numpy()).float()

def load_psd_label_pair(id='A1-1', condition='Vehicle'):
    return (load_psd(id,condition),load_one_hot_labels(id,condition))

def load_psd_label_pair_windowed(id='A1-1', condition='Vehicle',windowsize=5):
        Xi = load_psd(id, condition)
        Xi = cat([zeros(windowsize//2,42),Xi,zeros(windowsize//2,42)])
        Xi = window_epoched_signal(Xi,windowsize=windowsize,zero_padding=False)
        return (Xi,load_one_hot_labels(id,condition))

def load_psd_label_pairs(ids):
    X_train = Tensor()
    y_train = Tensor()
    for id in ids:
        for condition in ['Vehicle', 'PF']:
            Xi,yi = load_psd_label_pair(id,condition)
            X_train = cat([X_train, Xi])
            y_train = cat([y_train, yi])
    return X_train, y_train

def load_psd_label_pairs_windowed(ids,windowsize=5):
    X_train = Tensor()
    y_train = Tensor()
    for id in ids:
        for condition in ['Vehicle','PF']:
            Xi,yi = load_psd_label_pair_windowed(id,condition,windowsize)
            X_train = cat([X_train,Xi])
            y_train = cat([y_train,yi])
    return X_train,y_train
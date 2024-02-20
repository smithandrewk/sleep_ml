from os.path import join
from torch.utils.data import Dataset
from torch import load
import torch
from lib.ekyn import *
from sklearn.model_selection import train_test_split
from torch import cat,zeros
# Jan 29
class EpochedDatasetAnimalIdentifier(torch.utils.data.Dataset):
    """
    Dataset for training w1 resnets with ekyn data
    """
    def __init__(self,idx,condition):
        X,y = load_ekyn_pt(idx=get_ekyn_ids()[idx],condition=condition)
        self.X = X
        self.y = y
        self.idx = idx
    def __len__(self):
        return len(self.y)-3

    def __getitem__(self, idx):
        return (self.X[idx:idx+4].flatten(),torch.nn.functional.one_hot(torch.tensor(self.idx),num_classes=6).float())
# Jan 19
class SequencedDataset(torch.utils.data.Dataset):
    def __init__(self,idx,condition,sequence_length):
        self.sequence_length = sequence_length
        X,y = load_ekyn_pt_robust_scaled(idx=idx,condition=condition)
        self.X = torch.cat([torch.zeros(sequence_length // 2,5000),X,torch.zeros(sequence_length // 2,5000)])
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx:idx+self.sequence_length],self.y[idx])

# Jan 19
class EpochedDataset(torch.utils.data.Dataset):
    """
    Dataset for training w1 resnets with ekyn data
    """
    def __init__(self,idx,condition):
        X,y = load_ekyn_pt_robust_scaled(idx=idx,condition=condition)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx:idx+1],self.y[idx])


class Windowset(Dataset):
    def __init__(self,X,y):
        self.X = cat([zeros(4,5000),X,zeros(4,5000)])
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx:idx+9].flatten(),self.y[idx])
class Subjectset(Dataset):
    def __init__(self,ids):
        subjects = [load_eeg_label_pair(id=id,condition=condition,zero_pad=True,windowsize=9) for id in ids for condition in ['Vehicle','PF']]
        self.Xs = [subject[0] for subject in subjects]
        self.ys = [subject[1] for subject in subjects]
        del subjects
    def __len__(self):
        return 8640*len(self.Xs)

    def __getitem__(self, idx):
        return (self.Xs[idx // 8640][(idx % 8640) :(idx % 8640) + 9].flatten(),self.ys[idx // 8640][idx % 8640])
class ShuffleSplitDataset(Dataset):
    def __init__(self,training=True) -> None:
        super().__init__()
        subjects = [load_eeg_label_pair(id=id,condition=condition,zero_pad=True,windowsize=9) for id in get_ekyn_ids() for condition in ['Vehicle','PF']]
        self.Xs = [subject[0] for subject in subjects]
        self.ys = [subject[1] for subject in subjects]
        del subjects
        self.train_idx,self.test_idx = train_test_split(range(276480),test_size=.2)
        self.training = training
    def __len__(self):
        return len(self.train_idx) if self.training else len(self.test_idx)
    def __getitem__(self, index):
        index = self.train_idx[index] if self.training else self.test_idx[index]
        return (self.Xs[index // 8640][(index % 8640) : (index % 8640) + 9].flatten(),self.ys[index // 8640][index % 8640])
    def train(self):
        self.training = True
    def dev(self):
        self.training = False
class SSDataset(Dataset):
    def __init__(self,Xs,ys,idx) -> None:
        super().__init__()
        self.Xs = Xs
        self.ys = ys
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, index):
        index = self.idx[index]
        return (self.Xs[index // 8640][(index % 8640) : (index % 8640) + 9].flatten(),self.ys[index // 8640][index % 8640])
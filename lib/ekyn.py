import os
from torch import load
from lib.env import DATA_PATH
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,ConcatDataset

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
    return load(f'{DATA_PATH}/pt_ekyn/{id}_{condition}.pt',weights_only=False)

def load_ekyn_pt_robust(id,condition,downsampled):
    if downsampled:
        return load(f'{DATA_PATH}/pt_ekyn_robust_50hz/{id}_{condition}.pt',weights_only=True)
    else:
        return load(f'{DATA_PATH}/pt_ekyn_robust/{id}_{condition}.pt',weights_only=True)

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

def get_epoched_dataloader_for_ids(ids=['A1-1'],batch_size=512,shuffle=True,robust=True,condition=None):
    if condition is not None:
        conditions = [condition]
    else:
        conditions = CONDITIONS
    return DataLoader(
            dataset=ConcatDataset(
            [EpochedDataset(id=id,condition=condition,robust=robust,downsampled=True) for id in ids for condition in conditions] 
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=1
        )

def get_epoched_dataloaders(batch_size=512,shuffle_train=True,shuffle_test=False,robust=True):
    ekyn_ids = get_ekyn_ids()

    train_ids,test_ids = train_test_split(ekyn_ids,test_size=.2,shuffle=True,random_state=0)

    trainloader = get_epoched_dataloader_for_ids(ids=train_ids,batch_size=batch_size,shuffle=shuffle_train,robust=robust)
    testloader = get_epoched_dataloader_for_ids(ids=test_ids,batch_size=batch_size,shuffle=shuffle_test,robust=robust)
    
    print('train_ids',train_ids)
    print('test_ids',test_ids)
    print('n ids',len(ekyn_ids))
    print(f'{len(trainloader)} training batches {len(testloader)} testing batches')
    print(f'{len(trainloader)*batch_size} training samples {len(testloader)*batch_size} testing samples')
    print(f'{len(trainloader)*batch_size*10/3600:.2f} training hours {len(testloader)*batch_size*10/3600:.2f} testing hours')
    return trainloader,testloader

def get_epoched_dataloaders_loo(batch_size=512,shuffle_train=True,shuffle_test=False,robust=True,fold=0):
    train_ids = get_ekyn_ids()
    test_ids = [train_ids[fold]]
    del train_ids[fold]
    print(train_ids,test_ids)

    trainloader = get_epoched_dataloader_for_ids(ids=train_ids,batch_size=batch_size,shuffle=shuffle_train,robust=robust)
    testloader = get_epoched_dataloader_for_ids(ids=test_ids,batch_size=batch_size,shuffle=shuffle_test,robust=robust)
    
    print('train_ids',train_ids)
    print('test_ids',test_ids)
    print(f'{len(trainloader)} training batches {len(testloader)} testing batches')
    print(f'{len(trainloader)*batch_size} training samples {len(testloader)*batch_size} testing samples')
    print(f'{len(trainloader)*batch_size*10/3600:.2f} training hours {len(testloader)*batch_size*10/3600:.2f} testing hours')
    return trainloader,testloader



class SequencedDatasetInMemory(torch.utils.data.Dataset):
    def __init__(self,id,condition,sequence_length,stride=1):
        self.sequence_length = sequence_length
        self.stride = stride
        self.id = id
        self.condition = condition
        self.X,self.y = load_ekyn_pt(id=id,condition=condition)
        self.num_features,self.num_classes = self.X.shape[1],self.y.shape[1]
        self.X = torch.cat([torch.zeros(self.sequence_length // 2, self.num_features), self.X, torch.zeros(sequence_length // 2, self.num_features)]).unsqueeze(1)
        self.y = torch.cat([torch.zeros(self.sequence_length // 2, self.num_classes), self.y, torch.zeros(sequence_length // 2, self.num_classes)])
    def __len__(self):
        return (self.X.shape[0] - self.sequence_length + 1) // self.stride
    def __getitem__(self,idx):
        idx = self.stride*idx + self.sequence_length // 2
        return self.X[idx-(self.sequence_length // 2):idx+(self.sequence_length // 2)+1],self.y[idx]
    
def get_sequenced_dataloader_for_ids(ids=['A1-1'],sequence_length=3,batch_size=512,shuffle=True,condition=None):
    if condition is not None:
        conditions = [condition]
    else:
        conditions = CONDITIONS
    return DataLoader(
            dataset=ConcatDataset(
            [SequencedDatasetInMemory(id=id,condition=condition,sequence_length=sequence_length) for id in ids for condition in conditions] 
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=1
        )

def get_sequenced_dataloaders(batch_size=512,sequence_length=3,shuffle_train=True,shuffle_test=False,training_stride=1):
    ekyn_ids = get_ekyn_ids()

    train_ids,test_ids = train_test_split(ekyn_ids,test_size=.2,shuffle=True,random_state=0)

    trainloader = get_sequenced_dataloader_for_ids(ids=train_ids,sequence_length=sequence_length,batch_size=batch_size,shuffle=shuffle_train)
    testloader = get_sequenced_dataloader_for_ids(ids=test_ids,sequence_length=sequence_length,batch_size=batch_size,shuffle=shuffle_test)

    return trainloader,testloader

def get_sequenced_dataloaders_loo(batch_size=512,sequence_length=3,shuffle_train=True,shuffle_test=False,training_stride=1,fold=0):
    train_ids = get_ekyn_ids()
    test_ids = [train_ids[fold]]
    del train_ids[fold]
    print(train_ids,test_ids)

    trainloader = get_sequenced_dataloader_for_ids(ids=train_ids,sequence_length=sequence_length,batch_size=batch_size,shuffle=shuffle_train)
    testloader = get_sequenced_dataloader_for_ids(ids=test_ids,sequence_length=sequence_length,batch_size=batch_size,shuffle=shuffle_test)

    return trainloader,testloader


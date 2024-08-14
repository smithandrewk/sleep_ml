import torch
from torch import nn
import copy
from tqdm import tqdm
from lib.ekyn import *
import torch
from torch.utils.data import Dataset
from lib.ekyn import get_dataloaders,get_ekyn_ids
from torch.utils.data import DataLoader

class Dumbledore(nn.Module):
    def __init__(self,encoder_experiment_name,sequence_length,hidden_size=16,num_layers=1,dropout=None,frozen_encoder=True) -> None:
        super().__init__()
        self.frozen = frozen_encoder
        self.sequence_length = sequence_length
        self.encoder = self.get_encoder(encoder_experiment_name)
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(in_features=hidden_size,out_features=3)
    def forward(self,x):
        x = x.flatten(0,1)
        x = self.encoder(x)
        x = x.reshape(-1,self.sequence_length,3)
        output, (hn, cn) = self.lstm(x)
        x = nn.functional.relu(output[:,-1])
        x = self.classifier(x)
        return x
    def get_encoder(self,encoder_experiment_name):
        state = torch.load(f'experiments/{encoder_experiment_name}/state.pt',map_location='cpu')
        encoder = copy.deepcopy(state['model'])
        encoder.load_state_dict(state['best_model_wts'])
        if self.frozen:
            print("Model is freezing encoder")
            for p in encoder.parameters():
                p.requires_grad = False
        return encoder
class SequencedDatasetv2(Dataset):
    def __init__(self, id, condition, sequence_length, stride=1):
        self.sequence_length = sequence_length
        # X, y = load_ekyn_pt_robust(id=id, condition=condition, downsampled=True)
        X,y = load_ekyn_pt(id=id,condition=condition)
        
        # Assuming X.shape is (num_samples, num_features) and y.shape is (num_samples, num_classes)
        num_features = X.shape[1]
        num_classes = y.shape[1]
        
        # Pad the sequence
        self.X = torch.cat([torch.zeros(sequence_length // 2, num_features), X, torch.zeros(sequence_length // 2, num_features)]).unsqueeze(1)
        self.y = torch.cat([torch.zeros(sequence_length // 2, num_classes), y, torch.zeros(sequence_length // 2, num_classes)])
        
        self.stride = stride
        self.sequences = []
        self.labels = []
        for i in range(0, len(self.y) - sequence_length, stride):
            self.sequences.append(self.X[i:i + sequence_length])
            self.labels.append(self.y[i + sequence_length // 2])
        
        self.sequences = torch.stack(self.sequences)
        self.labels = torch.stack(self.labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def get_dataloaders(batch_size=512,shuffle_train=True,shuffle_test=False,sequence_length=3,stride=1):
    from sklearn.model_selection import train_test_split
    ekyn_ids = get_ekyn_ids()

    train_ids,test_ids = train_test_split(ekyn_ids,test_size=.2,shuffle=True,random_state=0)

    from torch.utils.data import DataLoader,ConcatDataset
    trainloader = DataLoader(
            dataset=ConcatDataset(
            [SequencedDatasetv2(id=id,condition=condition,sequence_length=sequence_length,stride=stride) for id in train_ids for condition in ['Vehicle','PF']] 
            ),
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=1
        )
    testloader = DataLoader(
            dataset=ConcatDataset(
            [SequencedDatasetv2(id=id,condition=condition,sequence_length=sequence_length,stride=1) for id in test_ids for condition in ['Vehicle','PF']] 
            ),
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=1
        )
    return trainloader,testloader

from sage.utils import count_params
sequence_length = 9
trainloader,testloader = get_dataloaders(batch_size=512,sequence_length=sequence_length,stride=50)
model = Dumbledore(encoder_experiment_name=f'2024_14_08_16_04_12',sequence_length=sequence_length,hidden_size=64,num_layers=1,dropout=.1,frozen_encoder=True)
print(count_params(model))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-2,weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
model = torch.compile(model)

trainlossi = []
testlossi = []
best_model_wts = copy.deepcopy(model.state_dict())
best_dev_loss = torch.inf
lossi = []

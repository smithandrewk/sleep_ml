from torch import nn
from torch.nn.functional import relu
import torch
from lib.env import *
class MLP(nn.Module):
    """
    MLP according to Wang et. al (proposed as 
    a baseline architecture for TSC)
    """
    def __init__(self,input_size=5000,hidden_sizes=(500,500,500)) -> None:
        super().__init__()
        self.d1 = nn.Dropout1d(p=.1)
        self.fc1 = nn.Linear(input_size,hidden_sizes[0])

        self.d2 = nn.Dropout1d(p=.2)
        self.fc2 = nn.Linear(hidden_sizes[0],hidden_sizes[1])

        self.d3 = nn.Dropout1d(p=.2)
        self.fc3 = nn.Linear(hidden_sizes[1],hidden_sizes[2])

        self.d4 = nn.Dropout1d(p=.3)
        self.fc4 = nn.Linear(hidden_sizes[2],3)

    def forward(self,x):
        x = self.d1(x)
        x = self.fc1(x)
        x = relu(x)

        x = self.d2(x)
        x = self.fc2(x)
        x = relu(x)

        x = self.d3(x)
        x = self.fc3(x)
        x = relu(x)

        x = self.d4(x)
        x = self.fc4(x)

        return x

class RecreatedMLPPSD(nn.Module):
    """
    MLP according to Smith et. al
    """
    def __init__(self,input_size=210) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size,512)

        self.fc2 = nn.Linear(512,512)

        self.fc3 = nn.Linear(512,512)

        self.fc4 = nn.Linear(512,3)

    def forward(self,x):
        x = self.fc1(x)
        x = relu(x)

        x = self.fc2(x)
        x = relu(x)

        x = self.fc3(x)
        x = relu(x)

        x = self.fc4(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self,in_feature_maps,out_feature_maps,n_features) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(in_feature_maps,out_feature_maps,kernel_size=8,padding='same',bias=False)
        self.bn1 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

        self.c2 = nn.Conv1d(out_feature_maps,out_feature_maps,kernel_size=5,padding='same',bias=False)
        self.bn2 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

        self.c3 = nn.Conv1d(out_feature_maps,out_feature_maps,kernel_size=3,padding='same',bias=False)
        self.bn3 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

        self.c4 = nn.Conv1d(in_feature_maps,out_feature_maps,1,padding='same',bias=False)
        self.bn4 = nn.LayerNorm((out_feature_maps,n_features),elementwise_affine=False)

    def forward(self,x):
        identity = x
        x = self.c1(x)
        x = self.bn1(x)
        x = relu(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = relu(x)

        x = self.c3(x)
        x = self.bn3(x)
        x = relu(x)

        identity = self.c4(identity)
        identity = self.bn4(identity)

        x = x+identity
        x = relu(x)
        
        return x

class Frodo(nn.Module):
    """
    the little wanderer
    """
    def __init__(self,n_features,device='cuda') -> None:
        super().__init__()
        self.n_features = n_features
        self.block1 = ResidualBlock(1,8,n_features).to(device)
        self.block2 = ResidualBlock(8,16,n_features).to(device)
        self.block3 = ResidualBlock(16,16,n_features).to(device)

        self.gap = nn.AvgPool1d(kernel_size=n_features)
        self.fc1 = nn.Linear(in_features=16,out_features=3)
    def forward(self,x,classification=True):
        x = x.view(-1,1,self.n_features)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        if(classification):
            x = self.fc1(x.squeeze())
            return x
        else:
            return x.squeeze()
        
class Gandalf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Frodo(n_features=5000,device=DEVICE).to(DEVICE)
        self.lstm = nn.LSTM(16,32,bidirectional=True)
        self.fc1 = nn.Linear(64,3)
    def forward(self,x_2d,classification=True):
        x_2d = x_2d.view(-1,9,1,5000)
        x = torch.Tensor().to(DEVICE)
        for t in range(x_2d.size(1)):
            xi = self.encoder(x_2d[:,t,:,:],classification=False)
            x = torch.cat([x,xi.unsqueeze(0)],dim=0)
        out,_ = self.lstm(x)
        if(classification):
            x = self.fc1(out[-1])
        else:
            x = out[-1]
        return x

class UniformRandomClassifier():
    def __init__(self) -> None:
        pass
    def fit(self,x,y):
        pass
    def predict(self,x):
        uniform_random_y_pred = torch.randint(0,3,(len(x),))
        return uniform_random_y_pred
    
class ProportionalRandomClassifier():
    def __init__(self) -> None:
        pass
    def fit(self,x,y):
        pass
    def predict(self,x):
        proportional_random_y_pred = torch.rand((len(x)))
        proportional_random_y_pred[proportional_random_y_pred <= .0613] = 2 # P
        proportional_random_y_pred[proportional_random_y_pred <= (.4558 + .0613)] = 4 # W
        proportional_random_y_pred[proportional_random_y_pred <= 1] = 3 # S
        proportional_random_y_pred = proportional_random_y_pred - 2
        proportional_random_y_pred = proportional_random_y_pred.long()
        return proportional_random_y_pred
class ResNetSmall(nn.Module):
    def __init__(self,n_features,device='cuda') -> None:
        super().__init__()
        self.n_features = n_features
        self.block1 = ResidualBlock(1,4,n_features).to(device)
        self.block2 = ResidualBlock(4,8,n_features).to(device)
        self.block3 = ResidualBlock(8,8,n_features).to(device)

        self.gap = nn.AvgPool1d(kernel_size=n_features)
        self.fc1 = nn.Linear(in_features=8,out_features=3)
    def forward(self,x,classification=True):
        x = x.view(-1,1,self.n_features)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        if(classification):
            x = self.fc1(x.squeeze())
            return x
        else:
            return x.squeeze()
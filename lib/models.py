from torch import nn
from torch.nn.functional import relu
import torch
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
class RawMLP(nn.Module):
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

class ResNet(nn.Module):
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
class CustomResNet(nn.Module):
    def __init__(self,n_features,device='cuda') -> None:
        super().__init__()
        self.n_features = n_features
        self.block1 = ResidualBlock(1,32,n_features).to(device)
        self.block2 = ResidualBlock(32,64,n_features).to(device)
        self.block3 = ResidualBlock(64,64,n_features).to(device)

        self.gap = nn.AvgPool1d(kernel_size=n_features)
        self.fc1 = nn.Linear(in_features=64,out_features=3)
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
class CNNLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = ResNet(5000).cuda()
        self.lstm = nn.LSTM(3,16)
        self.fc1 = nn.Linear(16,3)
    def forward(self,x_2d):
        x_2d = x_2d.view(-1,3,1,5000)
        x = torch.Tensor().cuda()
        for t in range(x_2d.size(1)):
            x_i = self.resnet(x_2d[:,t,:,:])
            x_i = x_i.view(-1,3)
            out,_ = self.lstm(x_i)
        x = self.fc1(out)        
        return x
class CNNBiLSTM(nn.Module):
    def __init__(self,device='cuda') -> None:
        super().__init__()
        self.resnet = ResNet().to(device)
        self.lstm_forward = nn.LSTM(3,128)
        self.lstm_backward = nn.LSTM(3,128)
        self.do1 = nn.Dropout(p=.2)
        self.fc1 = nn.Linear(256,3)
    def forward(self,x_2d):
        x_2d = x_2d.view(-1,9,1,5000)
        for t in range(5):
            x_i = self.resnet(x_2d[:,t,:,:])
            x_i = x_i.view(-1,3)
            f,_ = self.lstm_forward(x_i)
        for t in range(5):
            x_i = self.resnet(x_2d[:,-t,:,:])
            x_i = x_i.view(-1,3)
            b,_ = self.lstm_backward(x_i)
        x = torch.cat([f,b],axis=1)
        x = self.do1(x)
        x = self.fc1(x)        
        return x

class BigPapa(nn.Module):
    def __init__(self,device='cuda') -> None:
        super().__init__()
        self.device = device
        self.resnet = ResNet(5000).to(device)
        self.lstm = nn.LSTM(input_size=16,hidden_size=8,batch_first=True,bidirectional=True)
        self.fc1 = nn.Linear(2*8*9,3)
    def forward(self,x):
        x_t = x.view(-1,9,1,5000)
        x = torch.Tensor().to(self.device)
        for t in range(x_t.size(1)):
            x = torch.cat([x,self.resnet(x_t[:,t,:,:],classification=False).unsqueeze(1)],dim=1)
        x,_ = self.lstm(x.view(-1,9,16))
        x = self.fc1(x.reshape(-1,2*8*9))
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
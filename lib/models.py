from torch import nn
from torch.nn.functional import relu
import torch
class MLP(nn.Module):
    """
    MLP according to Wang et. al (proposed as 
    a baseline architecture for TSC)
    """
    def __init__(self,input_size=5000) -> None:
        super().__init__()
        self.d1 = nn.Dropout1d(p=.1)
        self.fc1 = nn.Linear(input_size,500)

        self.d2 = nn.Dropout1d(p=.2)
        self.fc2 = nn.Linear(500,500)

        self.d3 = nn.Dropout1d(p=.2)
        self.fc3 = nn.Linear(500,500)

        self.d4 = nn.Dropout1d(p=.3)
        self.fc4 = nn.Linear(500,3)

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
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(1, 10, kernel_size=10, stride=1,bias=False)
        self.c2 = nn.Conv1d(10, 10, kernel_size=8, stride=1,bias=False)
        self.c3 = nn.Conv1d(10, 10, kernel_size=5, stride=1,bias=False)
        self.fc1 = nn.Linear(6200,3)
        self.do1 = nn.Dropout(p=0.9)

    def forward(self,x):
        x = x.view(-1,1,5000)

        x = self.c1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x,kernel_size=2)

        x = self.c2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x,kernel_size=2)

        x = self.c3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x,kernel_size=2)

        x = self.fc1(x.view(-1,6200))
        x = self.do1(x)

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
        x,_ = self.lstm(x)
        x = self.fc1(x.reshape(-1,2*8*9))
        return x
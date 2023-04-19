from torch import nn
from torch.nn.functional import relu
import torch
## w1 model
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5000,10)
        self.do1 = nn.Dropout(p=0.9)
        self.fc2 = nn.Linear(10,3)
    def forward(self,x):
        x = self.fc1(x)
        x = relu(x)
        x = self.do1(x)
        x = self.fc2(x)
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
# class ResNet(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         ## block 1
#         n_feature_maps = 32
#         self.c1 = nn.Conv1d(1,n_feature_maps,kernel_size=8,padding='same',bias=False)
#         self.bn1 = nn.BatchNorm1d(n_feature_maps,momentum=.01)

#         self.c2 = nn.Conv1d(n_feature_maps,n_feature_maps,kernel_size=5,padding='same',bias=False)
#         self.bn2 = nn.BatchNorm1d(n_feature_maps,momentum=.01)

#         self.c4 = nn.Conv1d(1,n_feature_maps,1,padding='same',bias=False)
#         self.bn4 = nn.BatchNorm1d(n_feature_maps,momentum=.01)

#         ## final
#         self.gap = nn.AvgPool1d(kernel_size=5000)
#         self.fc1 = nn.Linear(in_features=n_feature_maps,out_features=3)

#     def forward(self,x):
#         x = x.view(-1,1,5000)
        
#         identity = x
#         x = self.c1(x)
#         x = self.bn1(x)
#         x = relu(x)

#         x = self.c2(x)
#         x = self.bn2(x)
#         x = relu(x)

#         identity = self.c4(identity)
#         identity = self.bn4(identity)

#         x = x+identity
#         x = relu(x)
        
#         ## final
#         x = self.gap(x)
#         x = self.fc1(x.squeeze())
        
#         return x
class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        ## block 1
        n_feature_maps = 64
        self.c1 = nn.Conv1d(1,n_feature_maps,kernel_size=8,padding='same',bias=False)
        self.bn1 = nn.LayerNorm((n_feature_maps,5000),elementwise_affine=False)

        self.c2 = nn.Conv1d(n_feature_maps,n_feature_maps,kernel_size=5,padding='same',bias=False)
        self.bn2 = nn.LayerNorm((n_feature_maps,5000),elementwise_affine=False)

        self.c4 = nn.Conv1d(1,n_feature_maps,1,padding='same',bias=False)
        self.bn4 = nn.LayerNorm((n_feature_maps,5000),elementwise_affine=False)

        ## final
        self.gap = nn.AvgPool1d(kernel_size=5000)
        self.fc1 = nn.Linear(in_features=n_feature_maps,out_features=3)

    def forward(self,x):
        x = x.view(-1,1,5000)
        
        identity = x
        x = self.c1(x)
        x = self.bn1(x)
        x = relu(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = relu(x)

        identity = self.c4(identity)
        identity = self.bn4(identity)

        x = x+identity
        x = relu(x)
        
        ## final
        x = self.gap(x)
        x = self.fc1(x.squeeze())
        
        return x
class CNNLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = ResNet().cuda()
        self.lstm = nn.LSTM(3,64)
        self.fc1 = nn.Linear(64,3)
    def forward(self,x_2d):
        x_2d = x_2d.view(-1,3,1,5000)
        x = torch.Tensor().cuda()
        for t in range(x_2d.size(1)):
            x_i = self.resnet(x_2d[:,t,:,:])
            x_i = x_i.view(-1,3)
            out,_ = self.lstm(x_i)

            # x = torch.cat([x,x_i.unsqueeze(0)])
        x = self.fc1(out)        
        # return x.view(-1,3,3)
        return x
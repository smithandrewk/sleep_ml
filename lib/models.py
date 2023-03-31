from torch import nn
from torch.nn.functional import relu

class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(1, 10, kernel_size=100, stride=1)
        self.fc1 = nn.Linear(2000, 256)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 3)


    def forward(self,x):
        x = self.c1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.do1(x)
        x = self.fc2(x)

        return x

class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        ## block 1
        n_feature_maps = 64
        self.c1 = nn.Conv1d(1,n_feature_maps,kernel_size=8,padding='same',bias=False)
        self.bn1 = nn.BatchNorm1d(n_feature_maps,momentum=.01)

        self.c2 = nn.Conv1d(n_feature_maps,n_feature_maps,kernel_size=5,padding='same',bias=False)
        self.bn2 = nn.BatchNorm1d(n_feature_maps,momentum=.01)

        self.c4 = nn.Conv1d(1,n_feature_maps,1,padding='same',bias=False)
        self.bn4 = nn.BatchNorm1d(n_feature_maps,momentum=.01)

        ## final
        self.gap = nn.AvgPool1d(kernel_size=500)
        self.fc1 = nn.Linear(in_features=n_feature_maps,out_features=3)

    def forward(self,x):
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
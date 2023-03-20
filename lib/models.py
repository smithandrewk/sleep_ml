from torch import nn
import torch
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=500,out_features=32)
        self.fc2 = nn.Linear(in_features=32,out_features=3)

    def forward(self,x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        
        return x

class CNN_0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(1, 16, kernel_size=100, stride=10, padding=1)
        self.c2 = nn.Conv1d(16, 8, kernel_size=100, stride=10, padding=1)
        self.fc1 = nn.Linear(56, 3)
        self.fc1.bias = torch.nn.Parameter(data=torch.Tensor([.1,1,1]))
        print(self.fc1.bias)

    def forward(self,x):
        x = self.c1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2)
        x = self.c2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2)
        x = x.view(-1, 56)
        x = self.fc1(x)
        
        return x
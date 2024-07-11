from torch import nn
import torch
from lib.env import *
from torch.nn.functional import relu

class ConvLayerNorm(nn.Module):
    def __init__(self,out_channels) -> None:
        super(ConvLayerNorm,self).__init__()
        self.ln = nn.LayerNorm(out_channels, elementwise_affine=False)

    def forward(self,x):
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        return x
class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.permute(*self.dims)
    
class ResBlockv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm='layer'):
        super(ResBlockv2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False)
        self.ln1 = ConvLayerNorm(out_channels) if norm == 'layer' else nn.BatchNorm1d(out_channels)
        # self.dropout = nn.Dropout(p=.1)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.ln2 = ConvLayerNorm(out_channels) if norm == 'layer' else nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                ConvLayerNorm(out_channels) if norm == 'layer' else nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.ln1(out)
        out = relu(out)
        out = self.dropout(out) 

        out = self.conv2(out)
        out = self.ln2(out)
        
        shortcut = self.shortcut(x)
        out += shortcut
        out = relu(out)
        return out
class ResNetv2(nn.Module):
    def __init__(self,block,widthi=[64],depthi=[2],n_output_neurons=3,norm='batch') -> None:
        super(ResNetv2, self).__init__()
        self.in_channels = widthi[0]
        self.stem = nn.Conv1d(1, widthi[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(widthi[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        layers = []
        for width,depth in zip(widthi,depthi):
            layers.append(self._make_layer(block=block,out_channels=width,blocks=depth,norm=norm))
        self.layers = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(widthi[-1], n_output_neurons)
    def _make_layer(self,block,out_channels,blocks,norm):
        layers = []
        layers.append(block(self.in_channels,out_channels,3,2,norm))
        self.in_channels = out_channels
        for _ in range(1,blocks):
            layers.append(block(out_channels,out_channels,3,1))
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layers(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
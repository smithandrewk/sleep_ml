from torch import nn
from torch.nn.functional import relu
import torch
from lib.env import *
import math
import json

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
        self.block1 = ResidualBlock(1,8,n_features)
        self.block2 = ResidualBlock(8,16,n_features)
        self.block3 = ResidualBlock(16,16,n_features)

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
        
class ResidualBlockv2(nn.Module):
    def __init__(self,n_features,in_feature_maps,out_feature_maps) -> None:
        super().__init__()
        self.in_feature_maps = in_feature_maps
        self.out_feature_maps = out_feature_maps
        if in_feature_maps != out_feature_maps:
            self.c1 = nn.Conv1d(in_channels=in_feature_maps,out_channels=out_feature_maps,kernel_size=3,stride=2,padding=1)
        else:
            self.c1 = nn.Conv1d(in_channels=in_feature_maps,out_channels=out_feature_maps,kernel_size=3,padding='same')
        self.ln1 = nn.LayerNorm(normalized_shape=(n_features))
        self.c2 = nn.Conv1d(in_channels=out_feature_maps,out_channels=out_feature_maps,kernel_size=3,padding='same')
        self.ln2 = nn.LayerNorm(normalized_shape=(n_features))

        self.downsample = nn.Conv1d(in_channels=in_feature_maps,out_channels=out_feature_maps,kernel_size=1,stride=2)
    def forward(self,x):
        identity = x
        x = self.c1(x)
        x = self.ln1(x)
        x = relu(x)
        x = self.c2(x)
        x = self.ln2(x)
        x = relu(x)
        if self.in_feature_maps != self.out_feature_maps:
            x = x + self.downsample(identity)
        else:
            x = x + identity
        return x

class ResNetv2(nn.Module):
    def __init__(self, windowsize, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.windowsize = windowsize * 5000
        starting_filters = 8

        self.c1 = nn.Conv1d(in_channels=1,out_channels=starting_filters,kernel_size=128,stride=2,padding=int(128//2) - 1)
        self.ln1 = nn.LayerNorm(normalized_shape=(int(self.windowsize/2)))
        self.mp1 = nn.MaxPool1d(kernel_size=2,stride=2)

        self.block1 = ResidualBlockv2(n_features=int(self.windowsize/4),in_feature_maps=starting_filters,out_feature_maps=starting_filters)        
        self.block2 = ResidualBlockv2(n_features=int(self.windowsize/4),in_feature_maps=starting_filters,out_feature_maps=starting_filters)        

        self.block3 = ResidualBlockv2(n_features=int(self.windowsize/8),in_feature_maps=starting_filters,out_feature_maps=starting_filters*2)        
        self.block4 = ResidualBlockv2(n_features=int(self.windowsize/8),in_feature_maps=starting_filters*2,out_feature_maps=starting_filters*2)        

        self.block5 = ResidualBlockv2(n_features=math.ceil(self.windowsize/16),in_feature_maps=starting_filters*2,out_feature_maps=starting_filters*4)        
        self.block6 = ResidualBlockv2(n_features=math.ceil(self.windowsize/16),in_feature_maps=starting_filters*4,out_feature_maps=starting_filters*4)   

        self.classifier = nn.Sequential(
            nn.AvgPool1d(kernel_size=math.ceil(self.windowsize/16)),
            nn.Flatten(start_dim=1),
            nn.Linear(starting_filters*4,3),
            # nn.ReLU(),
            # nn.Linear(32,3)
        )
    def forward(self,x):
        x = x.reshape(-1,1,self.windowsize)
        x = self.c1(x)
        x = self.ln1(x)
        x = relu(x)
        x = self.mp1(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)

        x = self.block5(x)
        x = self.block6(x)

        x = self.classifier(x)
        return x
    
class ResNetv3(nn.Module):
    def __init__(self, windowsize=1, starting_filters=8, n_blocks=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.windowsize = windowsize * 5000
        self.starting_filters = starting_filters
        
        self.c1 = nn.Conv1d(in_channels=1,out_channels=starting_filters,kernel_size=10,stride=2,padding=4)
        self.ln1 = nn.LayerNorm(normalized_shape=(int(self.windowsize/2)))
        self.mp1 = nn.MaxPool1d(kernel_size=2,stride=2)

        blocks = [
            ResidualBlockv2(n_features=math.ceil(self.windowsize/4),in_feature_maps=starting_filters,out_feature_maps=starting_filters),
            ResidualBlockv2(n_features=math.ceil(self.windowsize/4),in_feature_maps=starting_filters,out_feature_maps=starting_filters)
        ]
        
        for i in range(1,n_blocks):
            blocks.append(ResidualBlockv2(n_features=math.ceil(self.windowsize/(2**(2+i))),in_feature_maps=starting_filters*(2**(i-1)),out_feature_maps=starting_filters*(2**(i))))
            blocks.append(ResidualBlockv2(n_features=math.ceil(self.windowsize/(2**(2+i))),in_feature_maps=starting_filters*(2**(i)),out_feature_maps=starting_filters*(2**(i))))

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AvgPool1d(kernel_size=math.ceil(self.windowsize/(2**(2+n_blocks-1)))),
            nn.Flatten(start_dim=1),
            nn.Linear(starting_filters*(2**(n_blocks-1)),3),
            # nn.ReLU(),
            # nn.Linear(32,3)
        )
    def forward(self,x):
        x = x.reshape(-1,1,self.windowsize)
        x = self.c1(x)
        x = self.ln1(x)
        x = relu(x)
        x = self.mp1(x)

        x = self.blocks(x)

        x = self.classifier(x)
        return x

class ResNetv4(nn.Module):
    def __init__(self, in_features, block_sizes=(4,4,4,8),*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_features = in_features * 5000
        self.block_sizes = block_sizes

        kernel_size = 10
        padding = 4
        out_features = math.floor(((self.in_features+2*padding-1*(kernel_size-1)-1))/2+1)
        
        self.c1 = nn.Conv1d(in_channels=1,out_channels=block_sizes[0],kernel_size=kernel_size,stride=2,padding=padding,dilation=1)
        self.ln1 = nn.LayerNorm(normalized_shape=(out_features))
        out_features = math.floor(((out_features-2)/2)+1)
        self.mp1 = nn.MaxPool1d(kernel_size=2,stride=2)

        blocks = [
            ResidualBlockv2(n_features=out_features,in_feature_maps=block_sizes[0],out_feature_maps=block_sizes[0])
        ]
        for i in range(len(block_sizes)-1):
            if block_sizes[i] != block_sizes[i+1]:
                out_features = math.floor((out_features-1)/2+1)
            block = ResidualBlockv2(n_features=out_features,in_feature_maps=block_sizes[i],out_feature_maps=block_sizes[i+1])
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AvgPool1d(kernel_size=out_features),
            nn.Flatten(start_dim=1),
            nn.Linear(block_sizes[-1],3),
            # nn.ReLU(),
            # nn.Linear(32,3)
        )
    def forward(self,x):
        x = x.reshape(-1,1,self.in_features)
        x = self.c1(x)
        x = self.ln1(x)
        x = relu(x)
        x = self.mp1(x)

        x = self.blocks(x)
        x = self.classifier(x)
        return x

class RegNet(nn.Module):
    def __init__(self, in_features, in_channels, depthi=[1,1,3,1], widthi=[2,4,16,32], n_classes=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.in_channels = in_channels
        self.n_classes = n_classes

        in_features = math.floor(((in_features-1))/2+1)
        self.c1 = nn.Conv1d(in_channels=self.in_channels,out_channels=widthi[0],kernel_size=3,stride=2,padding=1,dilation=1)
        self.ln1 = nn.LayerNorm(normalized_shape=(in_features))
        self.mp1 = nn.MaxPool1d(kernel_size=2,stride=2)
        blocks = []
        in_feature_maps = widthi[0]
        for stage,depth in enumerate(depthi):
            for i in range(depth):
                if i == 0 and stage == 0:
                    in_features = math.floor(((in_features-2)/2)+1)
                    block = ResidualBlockv2(n_features=in_features,in_feature_maps=in_feature_maps,out_feature_maps=widthi[stage])
                    in_feature_maps = widthi[stage]
                elif i == 0:
                    in_features = math.floor(((in_features-1))/2+1)
                    block = ResidualBlockv2(n_features=in_features,in_feature_maps=in_feature_maps,out_feature_maps=widthi[stage])
                    in_feature_maps = widthi[stage]
                else:
                    block = ResidualBlockv2(n_features=in_features,in_feature_maps=widthi[stage],out_feature_maps=widthi[stage])
                blocks.append(block)
            
        self.blocks = nn.Sequential(*blocks)
        
        self.classifier = nn.Sequential(
            nn.AvgPool1d(kernel_size=in_features),
            nn.Flatten(start_dim=1),
            nn.Linear(widthi[-1],self.n_classes),
        )
    def forward(self,x,return_encoding=False):
        x = x.reshape(-1,self.in_channels,self.in_features)
        x = self.c1(x)
        x = self.ln1(x)
        x = relu(x)
        x = self.mp1(x)
        x = self.blocks(x)
        if return_encoding:
            return self.classifier[:2](x)
        x = self.classifier(x)
        return x

class Dumbledore(nn.Module):
    def __init__(self,encoder_path,sequence_length) -> None:
        super().__init__()
        self.encoder_path = encoder_path
        self.sequence_length = sequence_length
        self.encoder = self.get_encoder()
        self.lstm = nn.LSTM(3,8,num_layers=1,bidirectional=True,batch_first=True)
        self.fc1 = nn.Linear(16,3)
    def forward(self,x):
        x = self.encoder(x,return_encoding=False)
        x = x.view(-1,self.sequence_length,3)
        o,(h,c) = self.lstm(x)
        x = self.fc1(o[:,-1])
        return x
    def get_encoder(self):
        with open(f'{self.encoder_path}/config.json') as f:
            ENCODER_CONFIG = json.load(f)
        encoder = RegNet(in_features=ENCODER_CONFIG['WINDOW_SIZE'],in_channels=1,depthi=ENCODER_CONFIG['DEPTHI'],widthi=ENCODER_CONFIG['WIDTHI'])
        encoder.load_state_dict(torch.load(f'{self.encoder_path}/best.f1.pt'))
        print("Model is freezing encoder")
        # for p in encoder.parameters():
        #         p.requires_grad = False
        return encoder

def get_padding(l,out_l,k,s,d=1):
    if l % 2 == 0:
        return ((out_l-1)*s - l + d*(k-1))//2 + 1
    return ((out_l-1)*s - l + d*(k-1) + 1)//2 

class XBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, seq_len, b, g, p_dropout=None):
        super().__init__()

        stride = 1
        self.seq_len = seq_len
        if out_channels > in_channels:
            stride = 2
            self.seq_len = seq_len // 2 if seq_len % 2 == 0 else seq_len // 2 + 1

        padding = get_padding(seq_len, self.seq_len, kernel_size, stride)
        
        self.c = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // b, kernel_size=1, padding=0),
            nn.LayerNorm((seq_len)),
            nn.ReLU(),
            nn.Conv1d(in_channels // b, in_channels // b, kernel_size=kernel_size, padding=padding, groups=g, stride=stride),
            nn.LayerNorm((self.seq_len)),
            nn.ReLU(),
            nn.Conv1d(in_channels // b, out_channels, kernel_size=1, padding=0),
            nn.LayerNorm((self.seq_len)),
            nn.ReLU()
        )
        if p_dropout is not None and p_dropout > 0:
            self.c.add_module('dropout', nn.Dropout(p=p_dropout))
        
        self.identity = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
            nn.LayerNorm((self.seq_len)),
            nn.ReLU()
        ) if out_channels > in_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.identity(x) + self.c(x))

class RegNetX(nn.Module):
    def __init__(self, winsize, in_channels, stem_out_c, d: list, w: list, b, g, p_dropout=None):
        """
            stem_out_c: out channels of stem before first stage
            d: tuple of num blocks in each stage
            w: tuple of num channels in each stage
        """        
        super().__init__()
        self.winsize = winsize
        self.in_channels = in_channels
        self.stem_out_c = stem_out_c
        self.n_stage = len(d)
        if len(w) != len(d):
            raise ValueError('d and w must have same length')
        
        self.d_str = '-'.join([str(di) for di in d])
        self.w_str = '-'.join([str(wi) for wi in w])
        self.g = g
        self.b = b
        self.p_dropout = p_dropout

        s = nn.Sequential()
        w = [stem_out_c] + list(w)

        stem_pre_ln = math.floor(((winsize-1))/2+1)
        stem_out_len = math.floor(((stem_pre_ln-3))/2+1)
        
        for i in range(self.n_stage):
            rs = nn.Sequential()
            for j in range(d[i]):
                rs.add_module(f'stage-{i}_block-{j}', XBlock(
                    w[i] if j==0 else w[i+1], 
                    w[i+1], 
                    kernel_size=3, 
                    seq_len=rs[-1].seq_len if j>0 else s[-1][-1].seq_len if i>0 else stem_out_len, 
                    b=b, 
                    g=g, 
                    p_dropout=p_dropout
                ))
            s.add_module(f'stage-{i}', rs)

        self.e = nn.Sequential(
            nn.Conv1d(in_channels, stem_out_c, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm((stem_pre_ln)),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU(),
            s
        )

        self.o = nn.Sequential(
            nn.AvgPool1d(kernel_size=rs[-1].seq_len if rs else stem_out_len), # Nxdims[-1]x1
            nn.Flatten(start_dim=1), # Nxdims[-1]
            nn.Linear(in_features=w[-1], out_features=3)
        )

    def forward(self, x, return_embedding=False):
        x = x.view(-1, self.in_channels, self.winsize)
        x = self.e(x)
        if return_embedding:
            return self.o[0:2](x)
        x = self.o(x)
        return x
class YBlock(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding='same',bias=False),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.block(x)
        return x
        
class RegNetY(nn.Module):
    def __init__(self,depth,width,stem_kernel_size) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=width[0],kernel_size=stem_kernel_size,stride=2,padding=stem_kernel_size//2,bias=False),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU()
        )

        self.body = nn.Sequential()
        for stage_i in range(len(width)):
            for block_i in range(depth[stage_i]):
                print(block_i)
                self.body.add_module(name=f'{stage_i}_{block_i}',module=YBlock(in_channels=width[stage_i],out_channels=width[stage_i]))

        self.classifier = nn.Sequential(
            nn.AvgPool1d(kernel_size=1250),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=width[-1],out_features=3)
        )

    def forward(self,x):
        x = x.view(-1,1,5000)
        x = self.stem(x)
        x = self.body(x)
        x = self.classifier(x)
        return x
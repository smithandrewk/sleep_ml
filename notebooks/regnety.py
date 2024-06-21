from lib.utils import *
from lib.models import *

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import os
import json
import datetime
from pandas import DataFrame
import random
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Training program')
parser.add_argument('-r','--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument('--random', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument('--weighted', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument('-o','--overwrite', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument("--epochs", type=int, default=2000,help="Number of training iterations")
parser.add_argument("--device", type=int, default=0,help="Cuda device to select")
parser.add_argument("--width",nargs='+', type=int, help="Number of blocks")
parser.add_argument("--depth",nargs='+', type=int, help="Depth of each stage")
parser.add_argument("--testsize", type=float, default=.25, help="Depth of each stage")
parser.add_argument("--batch", type=int, default=256, help="Depth of each stage")
parser.add_argument("--lr", type=float, default=3e-3, help="Depth of each stage")
args = parser.parse_args()

DATE = datetime.datetime.now().strftime("%Y-%d-%m_%H:%M")
RESUME = args.resume
OVERWRITE = args.overwrite
EPOCHS = args.epochs
DEVICE_ID = args.device

CONFIG = {
    'DEVICE':'cuda',
    'PATIENCE':500,
    'PROGRESS':0,
    'DEPTHI':args.depth,
    'WIDTHI':args.width,
    'BEST_DEV_LOSS':torch.inf,
    'BEST_DEV_F1':0,
    'LAST_EPOCH':0,
    'TRAINLOSSI':[],
    'DEVLOSSI':[],
    'TRAINF1':[],
    'DEVF1':[],
    'BATCH_SIZE':args.batch,
    'LEARNING_RATE':args.lr,
    'TEST_SIZE':args.testsize,
    'STEM_KERNEL_SIZE':3,
    'WEIGHTED_LOSS':args.weighted
}
if CONFIG['DEVICE'] == 'cuda':
    DEVICE = f'{CONFIG["DEVICE"]}:{DEVICE_ID}'

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


print(train_idx,test_idx)
trainloader = DataLoader(ConcatDataset([EpochedDataset(idx=idx,condition=condition) for idx in train_idx for condition in ['Vehicle','PF']]),batch_size=CONFIG['BATCH_SIZE'],shuffle=True)
devloader = DataLoader(ConcatDataset([EpochedDataset(idx=idx,condition=condition) for idx in test_idx for condition in ['Vehicle','PF']]),batch_size=CONFIG['BATCH_SIZE'],shuffle=True)
from lib.models import RegNetY
model = RegNetY(depth=CONFIG['DEPTHI'],width=CONFIG['WIDTHI'],stem_kernel_size=CONFIG['STEM_KERNEL_SIZE'])
if CONFIG['WEIGHTED_LOSS']:
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([18.3846,  2.2810,  1.9716]))
else:
    criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=CONFIG['LEARNING_RATE'])
## End Your Code Here ##

CONFIG['PARAMS'] = sum([p.flatten().size()[0] for p in list(model.parameters())])
CONFIG['MODEL'] = str(model)
model.to(DEVICE)
criterion.to(DEVICE)

## Resume functionality
os.makedirs(f'{PROJECT_DIR}/{DATE}')
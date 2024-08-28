# TODO: add best model epoch
from lib.ekyn import get_epoched_dataloaders,get_epoched_dataloaders_loo,get_epoched_dataloaders_shuffle_split
from sage.utils import *
from sage.models import *
from lib.env import *
from time import time
import datetime
import copy
import os
import argparse

parser = argparse.ArgumentParser(description='Training program')
parser.add_argument("--device", type=int, default=0,help="Cuda device to select")
parser.add_argument("--batch", type=int, default=512,help="Batch Size")
args = parser.parse_args()

hyperparameters = {
    'experiment_group_id':'chillin',
    'wd':1e-2,
    'lr':3e-4,
    'batch_size':args.batch,
    'robust':False,
    'norm':'layer',
    'dropout':.5,
    'stem_kernel_size':3,
    'widthi':[64,128,256,512],
    'depthi':[2,2,2,2],
    'patience':100,
    'epochs':500,
    'device':f'cuda:{args.device}',
    'fold':0
}

# trainloader,testloader = get_epoched_dataloaders_loo(batch_size=hyperparameters['batch_size'],robust=hyperparameters['robust'],fold=hyperparameters['fold'])
# trainloader,testloader = get_epoched_dataloaders(
#     batch_size=hyperparameters['batch_size'],
#     robust=hyperparameters['robust']
#     )

trainloader,testloader = get_epoched_dataloaders_shuffle_split(
    batch_size=hyperparameters['batch_size'],
    test_size=.2
    )

model = ResNetv2(
    ResBlockv2,
    widthi=hyperparameters['widthi'],
    depthi=hyperparameters['depthi'],
    n_output_neurons=3,
    norm=hyperparameters['norm'],
    stem_kernel_size=hyperparameters['stem_kernel_size'],
    dropout=hyperparameters['dropout']
    )
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=hyperparameters['lr'],
    weight_decay=hyperparameters['wd']
    )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min',
    factor=0.5,
    patience=50
    )

state = {
    'start_time':datetime.datetime.now().strftime("%Y_%d_%m_%H_%M_%S"),
    'execution_time':0,
    'trainlossi':[],
    'testlossi':[],
    'best_dev_loss':torch.inf,
    'model':model,
    'scheduler':scheduler,
    'criterion':criterion,
    'optimizer':optimizer,
    'best_model_wts':copy.deepcopy(model.state_dict()),
}

for key in hyperparameters:
    state[key] = hyperparameters[key]

last_time = time()

os.makedirs(f'{EXPERIMENTS_PATH}/{state["start_time"]}')

for state in train(state,trainloader,testloader):
    plot_loss(state,EXPERIMENTS_PATH)
    state['execution_time'] = (state['execution_time'] + (time() - last_time))/2
    last_time = time()
    torch.save(state, f'{EXPERIMENTS_PATH}/{state["start_time"]}/state.pt')
torch.save(state, f'{EXPERIMENTS_PATH}/{state["start_time"]}/state.pt')

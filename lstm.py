# TODO : generator
# TODO : add patience to pbar
from lib.ekyn import *
from lib.env import *
from sage.utils import *
from sage.models import *
from time import time
import datetime
import copy
import os

experiment_group_id = 'lstm'
hyperparameters = {
    'wd':1e-2,
    'lr':3e-4,
    'batch_size':512,
    'norm':'batch',
    'dropout':.1,
    'patience':25,
    'scheduler_patience':20,
    'epochs':500,
    'sequence_length':9,
    'encoder_experiment_name':f'{EXPERIMENTS_PATH}/2024_29_08_13_04_17',
    'hidden_size':32,
    'num_layers':1,
    'fold':0,
    'robust':True,
    'bidirectional':False,
    'frozen_encoder':True,
    'use_embedding':True,
    'device':'cuda:0'
}

trainloader,testloader = get_sequenced_dataloaders_loo(**hyperparameters)
model = Dumbledore(**hyperparameters)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=hyperparameters['lr'],weight_decay=hyperparameters['wd'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=hyperparameters['scheduler_patience'])

state = {
    'start_time':datetime.datetime.now().strftime("%Y_%d_%m_%H_%M_%S"),
    'experiment_group_id':experiment_group_id,
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

print(state['start_time'])
print(count_params(model))
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
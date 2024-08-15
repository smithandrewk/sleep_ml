# TODO : generator
from lib.ekyn import *
from sage.utils import *
from sage.models import *
from time import time
import datetime
import copy
import os
from sage.models import Dumbledore
from lib.ekyn import *
from sage.utils import count_params

experiment_group_id = 'lstm'
hyperparameters = {
    'wd':1e-2,
    'lr':3e-4,
    'batch_size':2048,
    'robust':False,
    'norm':'batch',
    'dropout':.1,
    'stem_kernel_size':3,
    'widthi':[4,8,16,32],
    'depthi':[1,1,1,1],
    'patience':100,
    'scheduler_patience':50,
    'epochs':500,
    'sequence_length':9,
    'training_stride':1,
    'encoder_experiment_name':f'2024_14_08_16_04_12',
    'hidden_size':64,
    'num_layers':1,
    'frozen_encoder':True
}

trainloader,testloader = get_sequenced_dataloaders(batch_size=hyperparameters['batch_size'],sequence_length=hyperparameters['sequence_length'],stride=hyperparameters['training_stride'])
model = Dumbledore(encoder_experiment_name=hyperparameters['encoder_experiment_name'],sequence_length=hyperparameters['sequence_length'],hidden_size=hyperparameters['hidden_size'],num_layers=hyperparameters['num_layers'],dropout=hyperparameters['dropout'],frozen_encoder=hyperparameters['frozen_encoder'])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=hyperparameters['lr'],weight_decay=hyperparameters['wd'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=hyperparameters['scheduler_patience'])

state = {
    'start_time':datetime.datetime.now().strftime("%Y_%d_%m_%H_%M_%S"),
    'experiment_group_id':experiment_group_id,
    'execution_time':0,
    'device':'cuda:1',
    'trainlossi':[],
    'testlossi':[],
    'best_dev_loss':torch.inf,
    'model':model,
    'scheduler':scheduler,
    'criterion':criterion,
    'optimizer':optimizer,
    'best_model_wts':copy.deepcopy(model.state_dict()),
}

os.makedirs(f'experiments/{state["start_time"]}')

for key in hyperparameters:
    state[key] = hyperparameters[key]

last_time = time()

for state in train(state,trainloader,testloader):
    plot_loss(state)
    state['execution_time'] = (state['execution_time'] + (time() - last_time))/2
    last_time = time()
    torch.save(state, f'experiments/{state["start_time"]}/state.pt')
torch.save(state, f'experiments/{state["start_time"]}/state.pt')
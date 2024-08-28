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
    'stem_kernel_size':3,
    'patience':25,
    'scheduler_patience':20,
    'epochs':500,
    'sequence_length':3,
    'encoder_experiment_name':f'2024_27_08_18_50_05',
    'hidden_size':4,
    'num_layers':1,
    'robust':False,
    'bidirectional':False,
    'frozen_encoder':True,
    'device':'cuda:0',
    'fold':0
}

# trainloader,testloader = get_sequenced_dataloaders_loo(
#     batch_size=hyperparameters['batch_size'],
#     sequence_length=hyperparameters['sequence_length'],
#     fold=hyperparameters['fold']
#     )
trainloader,testloader = get_sequenced_dataloaders(
    batch_size=hyperparameters['batch_size'],
    sequence_length=hyperparameters['sequence_length'],
    )

model = Dumbledore(
    encoder_experiment_name=f'{EXPERIMENTS_PATH}/{hyperparameters["encoder_experiment_name"]}',
    sequence_length=hyperparameters['sequence_length'],
    hidden_size=hyperparameters['hidden_size'],
    num_layers=hyperparameters['num_layers'],
    dropout=hyperparameters['dropout'],
    frozen_encoder=hyperparameters['frozen_encoder'],
    bidirectional=hyperparameters['bidirectional']
    )

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

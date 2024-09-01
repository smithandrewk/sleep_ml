from lib.ekyn import *
from lib.env import *
from sage.utils import *
from sage.models import *
import datetime
import copy
import os

for encoder in [25]:
    hyperparameters = {
        'wd':1e-2,
        'lr':3e-4,
        'batch_size':512,
        'dropout':.1,
        'patience':25,
        'scheduler_patience':20,
        'epochs':500,
        'sequence_length':7,
        'encoder_experiment_name':f'{TMP_EXPERIMENTS_PATH}/{encoder}',
        'hidden_size':32,
        'num_layers':1,
        'fold':0,
        'dataloaders':'leave_one_out_sequenced',
        'robust':True,
        'bidirectional':False,
        'dev_set':True,
        'frozen_encoder':True,
        'device':'cuda:0'
    }

    dataloaders = get_dataloaders(**hyperparameters)
    model = Dumbledore(**hyperparameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=hyperparameters['lr'],weight_decay=hyperparameters['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=hyperparameters['scheduler_patience'])

    state = {
        'epoch':0,
        'start_time':datetime.datetime.now().strftime("%Y_%d_%m_%H_%M_%S"),
        'execution_time':0,
        'trainlossi':[],
        'devlossi':[],
        'devf1i':[],
        'testlossi':[],
        'testf1i':[],
        'best_dev_loss':torch.inf,
        'best_test_loss':torch.inf,
        'model':model,
        'scheduler':scheduler,
        'criterion':criterion,
        'optimizer':optimizer,
        'best_model_wts_dev_loss':copy.deepcopy(model.state_dict()),
        'best_model_wts_test_loss':copy.deepcopy(model.state_dict()),
    }

    for key in hyperparameters:
        state[key] = hyperparameters[key]

    experiment_path = f'{TMP_EXPERIMENTS_PATH}/{sorted([int(dir) for dir in os.listdir(f"{TMP_EXPERIMENTS_PATH}")])[-1] + 1}'

    os.makedirs(experiment_path)

    for state in train(state,**dataloaders):
        state['model'].eval()

        ## Temporary ##
        with torch.no_grad():
            loss,y_true,y_pred = evaluate(dataloader=dataloaders['testloader'],model=state['model'],criterion=state['criterion'],device=state['device'])
            state['testlossi'].append(loss)
            state['testf1i'].append(f1_score(y_true,y_pred,average='macro'))
            
            if state['testlossi'][-1] < state['best_test_loss']:
                state['best_test_loss'] = state['testlossi'][-1]
                state['best_test_loss_epoch'] = len(state['testlossi'])-1
                state['best_model_wts_test_loss'] = copy.deepcopy(state['model'].state_dict())

        ####################
        
        plot_loss(state,experiment_path)
        torch.save(state, f'{experiment_path}/state.pt')
    torch.save(state, f'{experiment_path}/state.pt')
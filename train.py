from lib.ekyn import *
from sage.utils import *
from sage.models import *
from lib.env import *
import datetime
import copy
import argparse

parser = argparse.ArgumentParser(description='Training program')
parser.add_argument("--device", type=int, default=0,help="Cuda Device")
parser.add_argument("--batch", type=int, default=512,help="Batch Size")
parser.add_argument("--fold", type=int, default=0,help="Testing Fold")
args = parser.parse_args()

fold = args.fold

for fold in range(16):
    for (widthi,depthi) in [
        ([64],[2]),
        ([64,128],[2,2]),
        ([64,128,256],[2,2,2]),
        ([64,128,256,512],[2,2,2,2])
        ]:
        hyperparameters = {
            'experiment_group_id':'encoder',
            'weight_decay':1e-2,
            'lr':3e-4,
            'batch_size':args.batch,
            'robust':True,
            'norm':'layer',
            'dropout':.1,
            'stem_kernel_size':3,
            'widthi':widthi,
            'depthi':depthi,
            'n_output_neurons':3,
            'patience':100,
            'epochs':500,
            'device':f'cuda:{args.device}',
            'dataloaders':'leave_one_out_epoched',
            'dev_set':True,
            'fold':fold
        }

        dataloaders = get_dataloaders(**hyperparameters)
        model = ResNetv2(block=ResBlockv2,**hyperparameters)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params=model.parameters(),lr=hyperparameters['lr'],weight_decay=hyperparameters['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

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

        # TODO : allow for multiple dataloaders
        # TODO : f1 score each epoch and add to plot_loss
        # TODO : add best model epoch
        # TODO : use a torch generator
        # TODO : remove things after yield in train

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
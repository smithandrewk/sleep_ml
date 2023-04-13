import torch
from tqdm import tqdm
from lib.utils import load_raw_list
import os
from sklearn.model_selection import train_test_split
normalize = True
balance = True
X,y = load_raw_list(range(32))

if(normalize):
    # center, stretch
    X = (X - X.mean(axis=1,keepdim=True))/X.std(axis=1,keepdim=True)
    # drop row if any element is inf
    not_inf_idx = torch.where(~X.isinf().any(axis=1))[0]
    X,y = X[not_inf_idx], y[not_inf_idx]

X = torch.cat([X[:-2],X[1:-1],X[2:]],axis=1)
y = y[2:]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,shuffle=True,random_state=0)

if (balance):
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=0)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train.argmax(axis=1))

from torch.nn.functional import one_hot
X_train_res = torch.from_numpy(X_train_res)
y_train_res_oh = one_hot(torch.from_numpy(y_train_res)).float()
data_dir = 'w3_small_balanced_normalized'
os.makedirs(data_dir,exist_ok=True)
os.makedirs(f'{data_dir}/train',exist_ok=True)
os.makedirs(f'{data_dir}/test',exist_ok=True)
torch.save(y_train_res_oh,f'{data_dir}/y_train.pt')
torch.save(y_test,f'{data_dir}/y_test.pt')
import torch
from tqdm import tqdm
from lib.utils import load_raw_list
import os
i = 0
for Xi in X_train_res:
    torch.save(Xi.clone(),f'{data_dir}/train/{i}.pt')
    i += 1
i = 0
for Xi in X_test:
    torch.save(Xi.clone(),f'{data_dir}/test/{i}.pt')
    i += 1
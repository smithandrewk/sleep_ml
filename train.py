# %%
from lib.ekyn import *
import matplotlib.pyplot as plt
from lib.models import MLP
from torch.utils.data import DataLoader,ConcatDataset
from lib.deep_learning_utils import count_params
import torch

# %%
ids = get_ekyn_ids()
conditions = ['Vehicle','PF']

# %%
from sklearn.model_selection import train_test_split
train_idx,test_idx = train_test_split(get_ekyn_ids(),test_size=.25,random_state=0)
train_idx,test_idx = train_idx,test_idx
print(train_idx,test_idx)

trainloader = DataLoader(
    dataset=ConcatDataset(
    [EpochedDataset(id=id,condition=condition) for id in train_idx for condition in conditions]
    ),
    batch_size=2048,
    shuffle=True
)
devloader = DataLoader(
    dataset=ConcatDataset(
    [EpochedDataset(id=id,condition=condition) for id in test_idx for condition in conditions]
    ),
    batch_size=2048,
    shuffle=False
)
print(len(trainloader)*2048)
print(len(devloader)*2048)

# %%
model = MLP(input_size=1000,hidden_sizes=(32,32,32))
model.cuda()
# model = torch.compile(model)
# torch.set_float32_matmul_precision('high')
print(f'{count_params(model)} parameters')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# %%
lossi = []
devlossi = []

# %%
lossi = []
trainlossi = []
trainf1 = []
devlossi = []
devf1 = []
model.train()
from tqdm import tqdm
from lib.deep_learning_utils import training_loop,development_loop
for i in tqdm(range(1000)):
    loss,f1 = training_loop(model=model,trainloader=trainloader,criterion=criterion,optimizer=optimizer,device='cuda')
    trainlossi.append(loss)
    trainf1.append(f1)

    loss,f1 = development_loop(model=model,devloader=devloader,criterion=criterion,device='cuda')
    devlossi.append(loss)
    devf1.append(f1)
    scheduler.step(loss)

    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(13,4),dpi=200)
    ax[0].plot(trainlossi)
    ax[0].plot(devlossi)
    ax[0].set_title(label='loss',fontweight='bold')
    ax[0].set_xlabel('Epoch')
    ax[1].plot(trainf1)
    ax[1].plot(devf1)
    ax[1].set_title(label='f1',fontweight='bold')
    ax[1].set_xlabel('Epoch')

    plt.savefig('loss.jpg')
    plt.close()

# %%
from lib.deep_learning_utils import evaluate
test_loss,report,y_true,y_pred,y_logits = evaluate(devloader,model,criterion)
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_true=y_true,y_pred=y_pred,normalize='true')
ConfusionMatrixDisplay.from_predictions(y_true=y_true,y_pred=y_pred,normalize='pred')



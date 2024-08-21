from random import seed, shuffle
from tqdm import tqdm
from lib.env import *
import os
import torch

def get_ids():
    return sorted([id.split("_")[0] for id in os.listdir(DATASET_PATH) if "PF" in id])

def get_leave_one_out_cv_ids_for_ekyn():
    ids = get_ids()
    seed(0)
    shuffle(ids)
    ret = []
    for test_id in ids:
        train_ids = [x for x in ids if x != test_id]
        ret.append((train_ids, [test_id]))
    return ret

def load_eeg_label_pair(id,condition):
    X,y = torch.load(f'{DATASET_PATH}/{id}_{condition}.pt',weights_only=False)
    X = torch.cat([torch.zeros(WINDOW_SIZE//2,5000),X,torch.zeros(WINDOW_SIZE//2,5000)])
    return (X,y)

def evaluate(dataloader,model,criterion,device='mps'):
    model.eval()
    model.to(device)
    with torch.no_grad():
        loss_total = 0
        y_true = []
        y_pred = []
        for Xi,yi in tqdm(dataloader):
            Xi,yi = Xi.to(device),yi.to(device)
            logits = model(Xi)
            loss = criterion(logits,yi)
            loss_total += loss.item()

            y_true.append(yi.argmax(axis=1).cpu())
            y_pred.append(logits.softmax(dim=1).argmax(axis=1).cpu())
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    return loss_total/len(dataloader),y_true,y_pred

class SSDataset(torch.utils.data.Dataset):
    def __init__(self,Xs,ys,idx) -> None:
        super().__init__()
        self.Xs = Xs
        self.ys = ys
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, index):
        index = self.idx[index]
        return (self.Xs[index // 8640][(index % 8640) : (index % 8640) + 9].flatten(),self.ys[index // 8640][index % 8640])

class Windowset(torch.utils.data.Dataset):
    def __init__(self,X,y):
        self.X = torch.cat([torch.zeros(4,5000),X,torch.zeros(4,5000)])
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx:idx+9].flatten(),self.y[idx])
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report,f1_score

def count_params(model):
    return sum([p.flatten().size()[0] for p in list(model.parameters())])

def evaluate(dataloader,model,criterion,device='cuda'):
    model.eval()
    model.to(device)
    from tqdm import tqdm
    with torch.no_grad():
        loss_total = 0
        y_true = []
        y_pred = []
        for Xi,yi in dataloader:
            Xi,yi = Xi.to(device),yi.to(device)
            logits = model(Xi)
            loss = criterion(logits,yi)
            loss_total += loss.item()

            y_true.append(yi.argmax(axis=1).cpu())
            y_pred.append(logits.softmax(dim=1).argmax(axis=1).cpu())
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    return loss_total / len(dataloader),y_true,y_pred

def training_loop(model,trainloader,criterion,optimizer,device):
    model.train()
    model.to(device)
    y_true = []
    y_pred = []
    loss_tr_total = 0
    for (X_tr,y_tr) in tqdm(trainloader,leave=False):
        y_tr = y_tr.reshape(-1,3)
        y_true.append(y_tr.argmax(axis=1).cpu())
        X_tr,y_tr = X_tr.to(device),y_tr.to(device)
        logits = model(X_tr)
        loss = criterion(logits,y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tr_total += loss.item()
        y_pred.append(torch.softmax(logits,dim=1).argmax(axis=1).detach().cpu())
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    return loss_tr_total/len(trainloader),f1_score(y_true,y_pred,average='macro')

def development_loop(model,devloader,criterion,device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        loss_dev_total = 0
        for (X_dv,y_dv) in tqdm(devloader,leave=False):
            y_dv = y_dv.reshape(-1,3)
            y_true.append(y_dv.argmax(axis=1).cpu())
            X_dv,y_dv = X_dv.to(device),y_dv.to(device)
            logits = model(X_dv)
            loss = criterion(logits,y_dv)
            loss_dev_total += loss.item()
            y_pred.append(torch.softmax(logits,dim=1).argmax(axis=1).detach().cpu())
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        return loss_dev_total/len(devloader),f1_score(y_true,y_pred,average='macro')
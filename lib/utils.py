import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from mne.io import read_raw_edf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset,DataLoader
from lib.models import *
import os
import sqlite3
from sqlite3 import Error
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix
import json
import os
from lib.env import *

def load_raw_edf_by_path(path):
    raw = read_raw_edf(path,verbose=False)
    raw.rename_channels({'EEG 1':'EEG','EEG 2':'EMG'})
    raw.set_channel_types({'EEG':'eeg','EMG':'emg'},verbose=False)
    return raw

def cm_grid(y_true,y_pred,save_path='cm.jpg'):
    fig,axes = plt.subplots(2,2,figsize=(5,5))
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='true'),annot=True,fmt='.2f',cbar=False,ax=axes[0][0])
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='pred'),annot=True,fmt='.2f',cbar=False,ax=axes[0][1])
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='all'),annot=True,fmt='.2f',cbar=False,ax=axes[1][0])
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred),annot=True,fmt='.0f',cbar=False,ax=axes[1][1])
    axes[0][0].set_title('Recall')
    axes[0][1].set_title('Precision')
    axes[1][0].set_title('Proportion')
    axes[1][1].set_title('Count')
    axes[0][0].set_xticks([])
    axes[0][1].set_xticks([])
    axes[0][1].set_yticks([])
    axes[1][1].set_yticks([])
    axes[0][0].set_yticklabels(['P','S','W'])
    axes[1][0].set_yticklabels(['P','S','W'])
    axes[1][0].set_xticklabels(['P','S','W'])
    axes[1][1].set_xticklabels(['P','S','W'])
    plt.savefig(save_path,dpi=200,bbox_inches='tight')

def metrics(y_true,y_pred):
    return {
        'precision':precision_score(y_true=y_true,y_pred=y_pred,average='macro'),
        'recall':recall_score(y_true=y_true,y_pred=y_pred,average='macro'),
        'f1':f1_score(y_true=y_true,y_pred=y_pred,average='macro')
    }

def evaluate(dataloader,model,criterion,DEVICE=DEVICE):
    with torch.no_grad():
        y_true = torch.Tensor()
        y_pred = torch.Tensor()
        y_logits = torch.Tensor()
        loss_total = 0
        for (Xi,yi) in dataloader:
            y_true = torch.cat([y_true,yi.argmax(axis=1)])

            Xi,yi = Xi.to(DEVICE),yi.to(DEVICE)
            logits = model(Xi)
            loss = criterion(logits,yi)
            loss_total += loss.item()
            
            y_logits = torch.cat([y_logits,torch.softmax(logits,dim=1).detach().cpu()])
            y_pred = torch.cat([y_pred,torch.softmax(logits,dim=1).argmax(axis=1).detach().cpu()])

    return loss_total/len(dataloader),metrics(y_true,y_pred),y_true,y_pred,y_logits

def window_epoched_signal(X,windowsize,zero_padding=True):
    """
    only works for odd windows, puts label at center
    """
    if(zero_padding):
        X = torch.cat([torch.zeros(windowsize//2,5000),X,torch.zeros(windowsize//2,5000)])
    cat = [X[:-(windowsize-1)]]
    for i in range(1,(windowsize-1)):
        cat.append(X[i:i-(windowsize-1)])
    cat.append(X[(windowsize-1):])
    X = torch.cat(cat,axis=1).float()
    return X

def make_cv_data_from_ekyn(foldi=0,window_size=1):
    train_size = .95
    data_dir = f'w{window_size}_cv_{foldi}'
    x_train_i = 0
    x_dev_i = 0
    normalize = False
    test_id,train_ids = get_leave_one_out_cv_ids_for_ekyn()[foldi]

    if(os.path.isdir(data_dir)):
        print(f'{data_dir} already exists')
        return
    os.makedirs(data_dir)
    os.makedirs(f'{data_dir}/train')
    os.makedirs(f'{data_dir}/dev')

    config = {
        'TRAIN_SIZE':train_size,
        'TRAIN_IDS':train_ids,
        'TEST_IDS':test_id,
        'NORMALIZED':normalize,
    }

    with open(f'{data_dir}/config.json', 'w') as f:
        f.write(json.dumps(config))

    y_train_all = torch.Tensor()
    y_dev_all = torch.Tensor()

    for id in tqdm(train_ids):
        for condition in ['PF','Vehicle']:
            X,y = load_eeg_label_pair(id=id,condition=condition)
            if(normalize):
                # center, stretch
                X = (X - X.mean(axis=1,keepdim=True))/X.std(axis=1,keepdim=True)
                # drop row if any element is inf
                not_inf_idx = torch.where(~X.isinf().any(axis=1))[0]
                X,y = X[not_inf_idx], y[not_inf_idx]
            # train test split for each file, approximates the same for train-test-splitting the entire set
            X_train,X_dev,y_train,y_dev = train_test_split(X,y,test_size=(1-train_size),shuffle=True,stratify=y,random_state=0)
            for Xi in X_train:
                torch.save(Xi.clone(),f'{data_dir}/train/{x_train_i}.pt')
                x_train_i += 1
            for Xi in X_dev:
                torch.save(Xi.clone(),f'{data_dir}/dev/{x_dev_i}.pt')
                x_dev_i += 1
            y_train_all = torch.cat([y_train_all,y_train])
            y_dev_all = torch.cat([y_dev_all,y_dev])

    torch.save(y_train_all,f'{data_dir}/y_train.pt')
    torch.save(y_dev_all,f'{data_dir}/y_dev.pt')

    y_test_all = torch.Tensor()
    os.makedirs(f'{data_dir}/test')
    x_test_i = 0

    for condition in ['PF','Vehicle']:
        X,y = load_eeg_label_pair(id=test_id,condition=condition)
        if(normalize):
            # center, stretch
            X = (X - X.mean(axis=1,keepdim=True))/X.std(axis=1,keepdim=True)
            # drop row if any element is inf
            not_inf_idx = torch.where(~X.isinf().any(axis=1))[0]
            X,y = X[not_inf_idx], y[not_inf_idx]
        for Xi in X:
            torch.save(Xi.clone(),f'{data_dir}/test/{x_test_i}.pt')
            x_test_i += 1
        y_test_all = torch.cat([y_test_all,y])
    torch.save(y_test_all,f'{data_dir}/y_test.pt')

def remove_outliers_from_eeg(eeg):
    from sklearn.impute import SimpleImputer
    eeg = eeg.reshape(-1,1)
    mean = np.mean(eeg)
    std = np.std(eeg,axis=0)

    lower_outliers = np.where(eeg < (mean - 5*std))[0]
    upper_outliers = np.where(eeg > (mean + 5*std))[0]

    eeg[lower_outliers] = np.nan
    eeg[upper_outliers] = np.nan

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    eeg_no_outliers = imp_mean.fit_transform(eeg)
    return eeg_no_outliers

def get_bout_statistics_for_predictions(pred):
    bout_lengths = {
    'P':[],
    'S':[],
    'W':[],
    'X':[],
    'A':[]
    }
    transition_matrix = pd.DataFrame(np.zeros((5,5)),columns=['P','S','W','X','A'],index=['P','S','W','X','A'])

    current_state = 'A'
    current_length = 0
    for epoch in pred:
        transition_matrix.loc[current_state,epoch] += 1
        if(epoch != current_state):
            bout_lengths[current_state].append(current_length)
            current_state = epoch
            current_length = 0
        current_length += 1
    bout_lengths[current_state].append(current_length)
    bout_lengths.pop('X')
    bout_lengths.pop('A')
    total = {key:sum(bout_lengths[key])*10/60 for key in bout_lengths}
    average = {key:np.mean(bout_lengths[key])*10 for key in bout_lengths}
    counts = {key:len(bout_lengths[key]) for key in bout_lengths}
    
    return pd.DataFrame([pd.Series(total,name='total'),pd.Series(average,name='average'),pd.Series(counts,name='counts')])



def zdb_logic(zdb_filename,csv_filename):
    try:
        conn = sqlite3.connect(zdb_filename)
    except Error as e:
        print(e)
    cur = conn.cursor()
    rename_dict = {'W':'Sleep-Wake', 'S':'Sleep-SWS', 'P':'Sleep-Paradoxical', 'X':''}
    offset = 10e7 # epoch time period
    # drop this table - creates issues
    query = "DROP TABLE IF EXISTS temporary_scoring_marker;"
    # get recordingstart
    query = "SELECT value FROM internal_property WHERE key='RecordingStart'"
    cur.execute(query)
    result = cur.fetchall()
    recording_start = float(result[0][0])
    recording_start = recording_start - (recording_start % 100000000) # get lower bound of current epoch
    cur.execute(query)
    #delete first score before adding machine data
    query = "DELETE FROM scoring_marker;"
    cur.execute(query)
    #delete first score before adding machine data
    query = "DELETE FROM scoring_revision;"
    cur.execute(query)
    query = f"""
        INSERT INTO scoring_revision 
        (id, name, is_deleted, tags, version, owner, date_created)
        VALUES 
        (1, 'LSTM', 0,'',0,'',{recording_start});
        """ 
    cur.execute(query)
    df = pd.read_csv(csv_filename)
    y_pred = df.to_numpy().squeeze()
    # insert new epochs with scoring into the table
    stop_time = recording_start
    for pred in y_pred:
        # calculate epoch
        start_time = stop_time
        stop_time = start_time+offset

        score = rename_dict[pred]
        # insert epoch
        query = f"""
                INSERT INTO scoring_marker 
                (starts_at, ends_at, notes, type, location, is_deleted, key_id)
                VALUES 
                ({start_time}, {stop_time}, '', '{score}', '', 0, 1);
                """ 
        cur.execute(query)
    conn.commit()
    conn.close()
def score_edf_lstm(id):
    device = 'cuda'
    model = BigPapa().to(device)
    model.load_state_dict(torch.load('../models/84.pt',map_location='cuda'))

    params = sum([p.flatten().size()[0] for p in list(model.parameters())])
    print("Params: ",params)
    eeg = load_raw_edf_by_path(f'../courtney_aug_oct_2022_baseline_recordings/1_raw_edf/{id}.edf').get_data(picks=['EEG'])

    X = torch.from_numpy(eeg.reshape(-1,5000)).float()
    del eeg
    # center, stretch
    X = (X - X.mean(axis=1,keepdim=True))/X.std(axis=1,keepdim=True)
    if(X.isinf().any()):
        print("inf")
    X = window_epoched_eeg(X,9)
    dataloader = DataLoader(TensorDataset(X),batch_size=16)
    y_pred = torch.Tensor().cuda()
    model.eval()
    for (X_test) in tqdm(dataloader):
        X_test = X_test[0].to(device)
        logits = model(X_test)
        y_pred = torch.cat([y_pred,torch.softmax(logits,dim=1).argmax(axis=1)])
    pred_expert = y_pred.cpu().numpy()
    for j in range(len(pred_expert)-2):
        if(pred_expert[j+1] != pred_expert[j] and pred_expert[j+1] != pred_expert[j+2]):
            pred_expert[j+1] = pred_expert[j]
    df = pd.DataFrame([pred_expert]).T
    df[df[0] == 0] = 'P'
    df[df[0] == 1] = 'S'
    df[df[0] == 2] = 'W'
    df = pd.concat([pd.DataFrame(np.array(['X']*4)),df,pd.DataFrame(np.array(['X']*4))]).reset_index(drop=True)
    if(not os.path.isdir(f'aging_pred')):
        os.system('mkdir aging_pred')
    df.to_csv(f'aging_pred/{id}.csv',index=False)
    return df
def score_edf_lstm_aging(fileindex):
    device = 'cuda'
    model = BigPapa().to(device)
    model.load_state_dict(torch.load('../models/84.pt',map_location='cuda'))

    params = sum([p.flatten().size()[0] for p in list(model.parameters())])
    print("Params: ",params)
    EEG_1 = [1,8,14,15,16]
    EEG_2 = [3,4,5,6,7,9,10,11,12,13,17]
    raw = load_raw_by_path(f'../data/courtney/1_raw_edf/23-May-{fileindex}.edf').get_data(picks=['EEG','EMG'])
    if(fileindex in EEG_1):
        eeg = raw[0]
    elif(fileindex in EEG_2):
        eeg = raw[1]
    else:
        raise Exception("fileindex not in index guide")
    X = torch.from_numpy(eeg.reshape(-1,5000)).float()
    del eeg
    # center, stretch
    X = (X - X.mean(axis=1,keepdim=True))/X.std(axis=1,keepdim=True)
    if(X.isinf().any()):
        print("inf")
    X = window_epoched_eeg(X,9)
    dataloader = DataLoader(TensorDataset(X),batch_size=16)
    y_pred = torch.Tensor().cuda()
    model.eval()
    for (X_test) in tqdm(dataloader):
        X_test = X_test[0].to(device)
        logits = model(X_test)
        y_pred = torch.cat([y_pred,torch.softmax(logits,dim=1).argmax(axis=1)])
    pred_expert = y_pred.cpu().numpy()
    for j in range(len(pred_expert)-2):
        if(pred_expert[j+1] != pred_expert[j] and pred_expert[j+1] != pred_expert[j+2]):
            pred_expert[j+1] = pred_expert[j]
    df = pd.DataFrame([pred_expert]).T
    df[df[0] == 0] = 'P'
    df[df[0] == 1] = 'S'
    df[df[0] == 2] = 'W'
    # TODO
    df = pd.concat([pd.DataFrame(np.array(['X']*4)),df,pd.DataFrame(np.array(['X']*4))]).reset_index(drop=True)
    if(not os.path.isdir(f'aging_pred')):
        os.system('mkdir aging_pred')
    df.to_csv(f'aging_pred/{fileindex}.csv',index=False)
    return df

def optimization_loop(model,trainloader,devloader,criterion,optimizer,epochs,DEVICE=DEVICE):
    loss_tr = []
    loss_dev = []

    pbar = tqdm(range(epochs))

    for epoch in pbar:
        loss_tr.append(training_loop(model,trainloader,criterion,optimizer,DEVICE))
        loss_dev.append(development_loop(model,devloader,criterion,DEVICE))

        pbar.set_description(f'\033[94m Train Loss: {loss_tr[-1]:.4f}\033[93m Dev Loss: {loss_dev[-1]:.4f}\033[0m')
        plt.plot(loss_tr[-20:])
        plt.plot(loss_dev[-20:])
        plt.savefig('running_loss.jpg')
        plt.close()

def training_loop(model,trainloader,criterion,optimizer,DEVICE):
    model.train()
    loss_tr_total = 0
    for (X_tr,y_tr) in trainloader:
        X_tr,y_tr = X_tr.to(DEVICE),y_tr.to(DEVICE)
        logits = model(X_tr)
        loss = criterion(logits,y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tr_total += loss.item()
    return loss_tr_total/len(trainloader)

def development_loop(model,devloader,criterion,DEVICE):
    model.eval()
    with torch.no_grad():
        loss_dev_total = 0
        for (X_dv,y_dv) in devloader:
            X_dv,y_dv = X_dv.to(DEVICE),y_dv.to(DEVICE)
            logits = model(X_dv)
            loss = criterion(logits,y_dv)
            loss_dev_total += loss.item()
        return loss_dev_total/len(devloader)
    
def optimization_loop_shuffle_split(model,dataloader,dataset,criterion,optimizer,epochs,DEVICE=DEVICE):
    loss_tr = []
    loss_dev = []

    pbar = tqdm(range(epochs))

    for epoch in pbar:
        dataset.train()
        loss_tr.append(training_loop(model,dataloader,criterion,optimizer,DEVICE))
        dataset.dev()
        loss_dev.append(development_loop(model,dataloader,criterion,DEVICE))

        pbar.set_description(f'\033[94m Train Loss: {loss_tr[-1]:.4f}\033[93m Dev Loss: {loss_dev[-1]:.4f}\033[0m')
        plt.plot(loss_tr[-20:])
        plt.plot(loss_dev[-20:])
        plt.savefig('running_loss.jpg')
        plt.close()
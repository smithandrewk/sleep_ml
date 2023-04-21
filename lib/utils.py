import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from mne.io import read_raw_edf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from torch import nn
from torch.nn.functional import relu,one_hot
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score,accuracy_score
def training_loss(train_dataloader,model,criterion,device):
    training_loss = 0
    for (X,y) in tqdm(train_dataloader):
        X,y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits,y)
        training_loss += loss.item()
    return training_loss/len(train_dataloader)
def cms(y_true,y_pred,path='.',loss=0):
    fig,axes = plt.subplots(1,3,sharey=True,figsize=(10,5))
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='true'),annot=True,ax=axes[0],cbar=False,fmt='.2f')
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='pred'),annot=True,ax=axes[1],cbar=False,fmt='.2f')
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred),annot=True,ax=axes[2],cbar=False,fmt='.2f')
    axes[0].set_title('Recall')
    axes[1].set_title('Precision')
    axes[2].set_title('Count')
    axes[0].set_xticklabels(['P','S','W'])
    axes[1].set_xticklabels(['P','S','W'])
    axes[2].set_xticklabels(['P','S','W'])
    axes[0].set_yticklabels(['P','S','W'])
    plt.suptitle(f'macro-recall : {balanced_accuracy_score(y_true=y_true,y_pred=y_pred)} loss : {loss}')
    plt.savefig(f'{path}/cm.jpg',dpi=200,bbox_inches='tight')
def load_raw(filename):
    filepath = f'../data/alpha_sleep/{filename}.edf'
    return load_raw_by_path(filepath)
def load_raw_list(list):
    ret = pd.DataFrame()

    raw = load_raw(list[0])
    df = load_psd(list[0])
    eeg = raw.get_data(picks='EEG')[0]
    X = pd.DataFrame(eeg.reshape(-1,5000))
    y = df['label']

    append = pd.concat([y,X],axis=1)
    ret = pd.concat([ret,append])
    for i in list[1:]:
        raw = load_raw(i)
        df = load_psd(i)
        eeg = raw.get_data(picks='EEG')[0]
        X = pd.DataFrame(eeg.reshape(-1,5000))
        y = df['label']

        append = pd.concat([y,X],axis=1)
        ret = pd.concat([ret,append])

    ret = ret.reset_index(drop=True)
    ret = ret[ret['label'] != 'X']

    y = np.array(pd.Categorical(ret.pop('label')).codes)
    X = ret.to_numpy()

    X = torch.from_numpy(X).float()
    y = one_hot(torch.from_numpy(y).long()).float()
    return (X,y)
def load_raw_by_path(path):
    raw = read_raw_edf(path,verbose=False)
    raw.rename_channels({'EEG 1':'EEG','EEG 2':'EMG'})
    raw.set_channel_types({'EEG':'eeg','EMG':'emg'},verbose=False)
    return raw

def load_psd(fileindex):
    df = pd.read_csv(f'../data/alpha_sleep/{fileindex}.csv')
    return df

def load_psd_list(list):
    df = load_psd(list[0])
    for i in list[1:]:
        df = pd.concat([df,load_psd(i)])
    df = df.reset_index(drop=True)
    return df

def load_all_psd():
    df = load_psd(0)
    for i in range(1,32):
        df = pd.concat([df,load_psd(i)])
    df = df.reset_index(drop=True)
    return df
def leave_one_out(left_out=0):
    df = pd.DataFrame()
    df_left_out = pd.DataFrame()

    for i in range(0,32):
        if(i==left_out):
            df_left_out = load_psd(i)
            continue
        df = pd.concat([df,load_psd(i)])
    df = df.reset_index(drop=True)
    return df,df_left_out
def leave_random_out(left_out=0):
    nums = np.arange(32)
    np.random.shuffle(nums)
    #wlog leave nums[0] out
    df_left_out = load_psd(nums[0])
    df = load_psd(nums[1])
    for i in nums[2:]:
        df = pd.concat([df,load_psd(i)])
    df = df.reset_index(drop=True)
    return df,df_left_out

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

def train_test_confusion_matrices(X_train,X_test,y_train,y_test,clf,title):
    fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(7.2,4.45),dpi=500,sharex=True,sharey=True)
    y_pred = clf.predict(X_train)
    cm = confusion_matrix(y_train,y_pred,normalize='true')
    sns.heatmap(ax=axes[0],data=cm,annot=True)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test,y_pred,normalize='true')
    sns.heatmap(ax=axes[1],data=cm,annot=True)
    fig.supxlabel('Predicted Label')
    fig.supylabel('True Label')
    axes[0].set_title(f'Training Data')
    axes[1].set_title(f'Testing Data')
    plt.suptitle(f'{title}',fontweight='heavy')
    plt.savefig(f'figures/{title}_train_test.jpg',dpi=200,bbox_inches='tight')
def test_confusion_matrix(X_test,y_test,clf,title):
    plt.figure(figsize=(5,5),dpi=500)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test,y_pred,normalize='true')
    sns.heatmap(data=cm,annot=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{title}',fontweight='heavy')
    plt.savefig(f'figures/{title}_test.jpg',dpi=200,bbox_inches='tight')

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
def test_evaluation(dataloader,model,criterion,device='cuda'):
    y_true = torch.Tensor()
    y_pred = torch.Tensor().to(device)
    model_was_training = False
    if(model.training):
        # note that this changes the state of the model outside the scope of this function
        model_was_training = True
        model.eval()

    loss_dev_total = 0
    for (X,y) in tqdm(dataloader):
        X,y = X.to(device),y.to(device)
        logits = model(X)
        loss = criterion(logits,y)
        loss_dev_total += loss.item()

        y_true = torch.cat([y_true,y.cpu().argmax(axis=1)])
        y_pred = torch.cat([y_pred,torch.softmax(logits,dim=1).argmax(axis=1)])

    cms(y_true=y_true,y_pred=y_pred.cpu(),loss=loss_dev_total/len(dataloader))

    if(model_was_training):
        model.train()

    return loss_dev_total/len(dataloader),y_true,y_pred
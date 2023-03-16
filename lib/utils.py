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

def load_raw(filename):
    filepath = f'data/{filename}.edf'
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
    y = torch.from_numpy(y).reshape(-1,1).long()
    return (X,y)
def load_raw_by_path(path):
    raw = read_raw_edf(path,verbose=False)
    raw.rename_channels({'EEG 1':'EEG','EEG 2':'EMG'})
    raw.set_channel_types({'EEG':'eeg','EMG':'emg'},verbose=False)
    # raw.set_channel_types({'Activity':'misc',
    #                     'EEG':'eeg',
    #                     'EMG':'emg',
    #                     'HD BattVoltage':'misc',
    #                     'On Time':'misc',
    #                     'Signal Strength':'misc',
    #                     'Temperature':'misc'},verbose=False)

    return raw

def load_psd(fileindex):
    df = pd.read_csv(f'data/{fileindex}.csv')
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
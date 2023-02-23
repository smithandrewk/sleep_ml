import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from mne.io import read_raw_edf

def load_raw(filename):
    filepath = f'data/{filename}.edf'
    return load_raw_by_path(filepath)

def load_raw_by_path(path):
    raw = read_raw_edf(path,verbose=False)
    raw.rename_channels({'EEG 1':'EEG','EEG 2':'EMG'})
    raw.set_channel_types({'Activity':'misc',
                        'EEG':'eeg',
                        'EMG':'emg',
                        'HD BattVoltage':'misc',
                        'On Time':'misc',
                        'Signal Strength':'misc',
                        'Temperature':'misc'})
    return raw

def load_psd(fileindex):
    df = pd.read_csv(f'data/{fileindex}.csv')
    for x,y in zip(df.columns[2:-2],np.linspace(0,19.5,40)):
        df = df.rename(mapper = {x:str(y)},axis=1)
    df = df.rename(mapper = {'EEG 2 (Mean, 10s)':'emg'},axis=1)
    df = df.rename(mapper = {'Activity (Mean, 10s)':'activity'},axis=1)
    df = df.drop('timestamp',axis=1)
    df = df[df['label']!='X']
    df = df.drop('0.0',axis=1)

    return df

def load_all_psd():
    df = load_psd(0)
    for i in range(1,32):
        df = pd.concat([df,load_psd(i)])
    df = df.reset_index(drop=True)
    return df

def leave_one_out():
    nums = np.arange(32)
    np.random.shuffle(nums)
    #wlog leave nums[0] out
    df_left_out = load_psd(nums[0])
    df = load_psd(nums[1])
    for i in nums[2:]:
        df = pd.concat([df,load_psd(i)])
    df = df.reset_index(drop=True)
    return df,df_left_out
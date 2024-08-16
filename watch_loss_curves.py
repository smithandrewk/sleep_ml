from lib.ekyn import *
from lib.env import *
from sage.utils import *
from sage.models import *
import pandas as pd
import os
import time

while True:
    states = {}
    for experiment in os.listdir(f'{EXPERIMENTS_PATH}'):
        if experiment == '.Trash-1000':
            continue
        state = torch.load(f'{EXPERIMENTS_PATH}/{experiment}/state.pt',map_location='cpu',weights_only=False)
        states[experiment] = state
    df = pd.DataFrame([states[experiment] for experiment in states])
    pd.set_option('display.max_rows', 500)
    df = df.sort_values(by='start_time',ascending=False)
    df = df.reset_index(drop=True)
    plot_loss_curves(df.iloc[[0,3]],moving_window_length=1,lstm=False)

    time.sleep(30) # seconds

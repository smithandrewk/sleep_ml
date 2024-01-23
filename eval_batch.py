import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

df = pd.DataFrame()
for project in os.listdir(f'projects'):
    if not os.path.exists(f'projects/{project}/config.json'):
        continue
    with open(f'projects/{project}/config.json','r') as f:
        CONFIG = json.load(f)
    df[project] = pd.Series(CONFIG)
df = df.T
df = df.sort_index()
print(df.sort_values(by='BEST_DEV_LOSS',ascending=True)[['BEST_DEV_LOSS','BEST_DEV_F1','DEPTHI','WIDTHI']])
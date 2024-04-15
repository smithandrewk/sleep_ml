import os
import matplotlib.pyplot as plt
import json
import pandas as pd
from termcolor import colored

pd.set_option('display.max_rows', 500)

dir = f'projects'
files = os.listdir(dir)
df = pd.DataFrame()
for project in files:
    if not os.path.exists(f'{dir}/{project}/config.json'):
        continue
    with open(f'{dir}/{project}/config.json','r') as f:
        CONFIG = json.load(f)
        CONFIG['PROJECT_NAME'] = project
    df_dictionary = pd.DataFrame([CONFIG],index=[project])
    df = pd.concat([df, df_dictionary])
if 'SEQUENCE_LENGTH' not in df.columns:
    print(df.set_index(['PROJECT_NAME']).sort_values(by='BEST_DEV_LOSS',ascending=True)[['BEST_DEV_LOSS','BEST_DEV_F1','DEPTHI','WIDTHI','ENCODER_PATH']])
else:
    print(df.set_index(['PROJECT_NAME']).sort_values(by='BEST_DEV_LOSS',ascending=True)[['BEST_DEV_LOSS','BEST_DEV_F1','DEPTHI','WIDTHI','PARAMS','SEQUENCE_LENGTH','HIDDEN_DIM','FROZEN','EMBEDDING','ENCODER_PATH','LAYERS','LEARNING_RATE']])
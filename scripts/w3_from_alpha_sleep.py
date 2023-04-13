"""
w3 unbalanced unnormalized
"""

import torch
from tqdm import tqdm
from lib.utils import load_raw_list
import os
i = 0
os.makedirs('w3')
for fileindex in tqdm(range(32)):
    X,y = load_raw_list([fileindex])
    X = torch.cat([X[:-2],X[1:-1],X[2:]],axis=1)
    y = y[1:-1]
    for (Xi,yi) in zip(X,y):
        torch.save((Xi.clone(),yi.clone()),f'w3/{i}.pt')
        i += 1
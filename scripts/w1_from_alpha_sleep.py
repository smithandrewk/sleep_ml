import torch
from tqdm import tqdm
from lib.utils import load_raw_list
import os
i = 0
os.makedirs('w1')
for fileindex in tqdm(range(32)):
    X,y = load_raw_list([fileindex])
    for (Xi,yi) in zip(X,y):
        torch.save((Xi.clone(),yi.clone()),f'w1/{i}.pt')
        i += 1
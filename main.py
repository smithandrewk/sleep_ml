#! /usr/bin/env python3
"""
author: Andrew Smith
date: December 12th, 2024
description: Reboot of gandalf training script in reviews for NPP
"""
from lib.utils import *
import torch
from lib.models import Gandalf
from datetime import datetime

fold = 0
lr = 3e-4
batch_size = 32
patience = 30
current_date = str(datetime.now()).replace(' ','_')
device = f'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"

model = Gandalf()
model.to(device);

trainloader,testloader = get_dataloaders_for_fold(fold)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
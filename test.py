# %%
from lib.utils import *
from lib.models import *
from lib.ekyn import *
from lib.env import *
from lib.datasets import *

# %%
class Subjectset(Dataset):
    def __init__(self,ids):
        subjects = [load_eeg_label_pair(id=id,condition=condition,zero_pad=True,windowsize=9) for id in ids for condition in ['Vehicle','PF']]
        self.Xs = [subject[0] for subject in subjects]
        self.ys = [subject[1] for subject in subjects]
        del subjects
    def __len__(self):
        return 8640*len(self.Xs)

    def __getitem__(self, idx):
        return (self.Xs[idx // 8640][(idx % 8640) :(idx % 8640) + 9].flatten(),self.ys[idx // 8640][idx % 8640])

# %%
trainloader = DataLoader(Subjectset(subjects=[load_eeg_label_pair(id=id,condition=condition,zero_pad=True,windowsize=9) for id in get_ekyn_ids() for condition in ['Vehicle','PF']]),batch_size=32,shuffle=True)

# %%
class A(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(45000,10)
        self.fc2 = nn.Linear(10,3)

    def forward(self,x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x
model = A().to(DEVICE)

# %%
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
params = sum([p.flatten().size()[0] for p in list(model.parameters())])
print("Params: ",params)

# %%
import time
start_time = time.time()
model.train()
loss_tr_total = 0
for (X_tr,y_tr) in tqdm(trainloader):
    X_tr,y_tr = X_tr.to(DEVICE),y_tr.to(DEVICE)
    logits = model(X_tr)
    loss = criterion(logits,y_tr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_tr_total += loss.item()
print("--- %s seconds ---" % (time.time() - start_time))



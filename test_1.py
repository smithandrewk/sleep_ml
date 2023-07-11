# %%
from lib.utils import *
from lib.models import *
from lib.ekyn import *
from lib.env import *
from lib.datasets import *

# %%
trainloader = DataLoader(dataset=ShuffleSplitDataset(),batch_size=32,shuffle=True)
devloader = DataLoader(dataset=ShuffleSplitDataset(training=False),batch_size=32,shuffle=True)

# %%
class MODEL(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(45000,3)
    def forward(self,x):
        return self.fc1(x)
model = MODEL().to(DEVICE)

# %%
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
params = sum([p.flatten().size()[0] for p in list(model.parameters())])
print("Params: ",params)

# %%
optimization_loop(model,trainloader,devloader,criterion,optimizer,epochs=20,DEVICE=DEVICE)


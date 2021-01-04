import torch.nn as nn

class Sequential(nn.Module):
    def __init__(self):
        super(Sequential,self).__init__()
        self.l1 = nn.Linear(28*28,100)
        self.l2 = nn.Linear(100,10)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)

        return x
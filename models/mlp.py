## predicts entire value map
## rather than a single value

import torch
import math, torch.nn as nn, pdb
import torch.nn.functional as F
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        
        layers = []

        for ind in range(len(sizes)-1):
            layers.append( nn.Linear(sizes[ind], sizes[ind+1]) )
            layers.append( nn.ReLU() )
        layers.pop(-1)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x

if __name__ == '__main__':
    batch_size = 32
    sizes = [10, 128, 128, 4]
    mlp = MLP(sizes)
    inp = Variable( torch.Tensor(batch_size, sizes[0]) )
    print inp.size()
    out = mlp(inp)
    print out.size()
    loss = out.sum()
    loss.backward()














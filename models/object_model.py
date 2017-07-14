import sys, math
import numpy as np
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim

'''
Object observations are single-channels images
with positive indices denoting objects.
0's denote no object.

'''

class ObjectModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, inp_size, out_dim):
        super(ObjectModel, self).__init__()

        self.reshape = [-1]
        for dim in inp_size:
            self.reshape.append(dim)
        self.reshape.append(embed_dim)

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv2d(embed_dim, 3, kernel_size=5)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv3 = nn.Conv2d(6,12, kernel_size=5)
        self.conv4 = nn.Conv2d(12,12, kernel_size=5)
        self.fc1 = nn.Linear(192, out_dim)

    def forward(self, x):
        x = x.view(-1)
        x = self.embed(x)
        x = x.view(*self.reshape)
        x = x.transpose(1,-1).squeeze()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 192)
        x = self.fc1(x)
        return x

if __name__ == '__main__':
    inp = torch.LongTensor(1,20,20).zero_()
    vocab_size = 10
    emb_dim = 3
    rank = 10
    phi = ObjectModel(vocab_size, emb_dim,inp.size(), rank)

    # enc = nn.Embedding(10,emb_dim,padding_idx=0)
    inp = torch.LongTensor(5,1,20,20).zero_()
    inp[0][0][0][0]=1
    # inp[0][1][0][0]=1
    inp[1][0][0][2]=1
    print inp
    inp = Variable(inp.view(-1))

    out = phi.forward(inp)
    # print out
    # out = out.view(-1,2,3,3,emb_dim)
    out = out.data
    print out.size()

    # print out[0][0][0]
    # print out[1][0][0]


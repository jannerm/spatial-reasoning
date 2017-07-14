import sys, math
import numpy as np
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim

'''
State observations are two-channel images
with 0: puddle, 1: grass, 2: agent

'''

class MapModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, out_dim):
        super(MapModel, self).__init__()

        self.embed_dim = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, 3, kernel_size=3)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3)
        self.conv3 = nn.Conv2d(6,12, kernel_size=3)
        # self.conv4 = nn.Conv2d(12,12, kernel_size=5)
        self.fc1 = nn.Linear(192, out_dim)

    def forward(self, x):
        reshape = []
        for dim in x.size(): reshape.append(dim) 
        reshape.append(self.embed_dim)
        
        ## reshape to vector
        x = x.view(-1)
        ## get embeddings
        x = self.embed(x)
        ## reshape to batch x channels x M x N x embed_dim
        x = x.view(*reshape)
        ## sum over channels in input
        x = x.sum(1)
        ## reshape to batch x embed_dim x M x N
        ## (treats embedding dims as channels)
        x = x.transpose(1,-1).squeeze()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 192)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    from torch.autograd import Variable
    # inp = torch.LongTensor(2,10,10).zero_()
    vocab_size = 10
    emb_dim = 3
    rank = 7
    phi = MapModel(vocab_size, emb_dim, rank)

    # enc = nn.Embedding(10,emb_dim,padding_idx=0)
    inp = torch.LongTensor(5,2,10,10).zero_()
    inp[0][0][0][0]=1
    # inp[0][1][0][0]=1
    inp[1][0][0][2]=1
    print inp.size()
    inp = Variable(inp)

    out = phi.forward(inp)
    # print out
    # out = out.view(-1,2,3,3,emb_dim)
    out = out.data
    print out.size()

    # print out[0][0][0]
    # print out[1][0][0]








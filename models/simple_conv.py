import sys, math, pdb
import numpy as np
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim

'''
State observations are two-channel images
with 0: puddle, 1: grass, 2: agent

'''

class SimpleConv(nn.Module):
    def __init__(self, in_channels):
        super(SimpleConv, self).__init__()

        self.in_channels = in_channels

        # self.embed = nn.Embedding(vocab_size, in_channels)
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3,  6, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(6, 12, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(12,18, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(18,24, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(24,18, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(18,12, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(12, 6, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(6,  3, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        # # self.conv4 = nn.Conv2d(12,12, kernel_size=5)
        # self.fc1 = nn.Linear(192, out_dim)

    def forward(self, x):
        # reshape = []
        # for dim in x.size(): reshape.append(dim) 
        # reshape.append(self.in_channels)

        # ## reshape to vector
        # x = x.view(-1)
        # ## get embeddings
        # x = self.embed(x)
        # ## reshape to batch x channels x M x N x embed_dim
        # x = x.view(*reshape)
        # ## sum over channels in input
        # x = x.sum(1)
        # # pdb.set_trace()
        # ## reshape to batch x embed_dim x M x N
        # ## (treats embedding dims as channels)
        # x = x.transpose(1,-1)[:,:,:,:,0] #.squeeze() #
        # print 'SIZE:', x.size()
        # pdb.set_trace()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        # x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.conv10(x)

        # x = x.view(-1, 192)
        # x = self.fc1(x)
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








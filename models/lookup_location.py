import sys, math, pdb
import numpy as np
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

'''
Object observations are single-channels images
with positive indices denoting objects.
0's denote no object.

'''

class LookupLocationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, map_dim = 10):
        super(LookupLocationModel, self).__init__()

        ## add two for (x,y) location channels
        self.embed_dim = embed_dim

        # self.reshape = [-1]
        # for dim in inp_size:
            # self.reshape.append(dim)
        # self.reshape.append(self.embed_dim)

        # self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # self.conv1 = nn.Conv2d(embed_dim, 3, kernel_size=5)
        # self.conv2 = nn.Conv2d(3, 6, kernel_size=5)
        # self.conv3 = nn.Conv2d(6,12, kernel_size=5)
        # self.conv4 = nn.Conv2d(12,12, kernel_size=5)
        # self.fc1 = nn.Linear(192, out_dim)
        # self.init_weights()

        self.map_dim = map_dim
        self.locations = self.__init_locations(self.map_dim)

    def __init_locations(self, map_dim):
        row = torch.arange(0,map_dim).unsqueeze(1).repeat(1,map_dim)
        col = torch.arange(0,map_dim).repeat(map_dim,1)
        locations = torch.stack( (row, col) )
        return Variable(locations.cuda())

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # print 'LOOKUP: ', x.size(), type(x)
        batch_size = x.size(0)

        reshape = []
        for dim in x.size(): reshape.append(dim) 
        reshape.append(self.embed_dim)

        if x.size(-1) != self.map_dim:
            self.map_dim = x.size(-1)
            self.locations = self.__init_locations(self.map_dim)

        ## reshape to vector
        x = x.view(-1)
        ## get embeddings
        x = self.embed(x)
        ## reshape to batch x channels x M x N x embed_dim
        x = x.view(*reshape)
        ## sum over channels in input
        x = x.sum(1)
        # pdb.set_trace()
        ## reshape to batch x embed_dim x M x N
        ## (treats embedding dims as channels)
        x = x.transpose(1,-1)[:,:,:,:,0] #.squeeze() #

        locations = self.locations.unsqueeze(0).repeat(batch_size,1,1,1)
        
        # pdb.set_trace()
        x = torch.cat( (x, locations), 1 )
        # print self.locations
        # print locations
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = x.view(-1, 192)
        # x = self.fc1(x)
        return x

if __name__ == '__main__':
    vocab_size = 10
    emb_dim = 3
    map_dim = 10
    phi = LookupLocationModel(vocab_size, emb_dim, map_dim=map_dim)

    # enc = nn.Embedding(10,emb_dim,padding_idx=0)
    inp = torch.LongTensor(5,1,map_dim,map_dim).zero_()
    inp[0][0][0][0]=1
    # inp[0][1][0][0]=1
    inp[1][0][0][2]=1
    # print inp
    inp = Variable(inp)

    out = phi.forward(inp)
    # print out
    # out = out.view(-1,2,3,3,emb_dim)
    out = out.data
    print out.size()

    # print out[0][0][0]
    # print out[1][0][0]


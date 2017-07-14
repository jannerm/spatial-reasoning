import sys, math
import numpy as np
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb

'''
Object observations are single-channels images
with positive indices denoting objects.
0's denote no object.

'''

class ObjectModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, inp_size, out_dim):
        super(ObjectModel, self).__init__()
        (self.num_objects, self.embed_dim) = inp_size
        self.hidden_dim_1 = 10
        self.hidden_dim_2 = 10
        self.out_dim = out_dim

        self.reshape = [-1]
        for dim in inp_size:
            self.reshape.append(dim)
        self.reshape.append(embed_dim)

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim + 2, self.hidden_dim_1)
        self.fc2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.fc3 = nn.Linear(self.hidden_dim_2, self.out_dim)
        # self.fc1 = nn.Linear(embed_dim+2, 20)
        # self.conv1 = nn.Conv2d(embed_dim, 3, kernel_size=5)
        # self.conv2 = nn.Conv2d(3, 6, kernel_size=5)
        # self.conv3 = nn.Conv2d(6,12, kernel_size=5)
        # self.conv4 = nn.Conv2d(12,12, kernel_size=5)
        # self.fc1 = nn.Linear(192, out_dim)
        self.fc_temp = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)

    ## batch x 5 x 3
    def forward(self, x):
        ## batch x 5 x 1
        indices = x[:,:,0]

         ## batch x 5 x embed
        embeddings = self.embed(indices)

        ## batch x 5 x pos
        positions = x[:,:,1:].float()

        ## join embeddings and positions
        ## batch x 5 x (embed + pos)
        x = torch.cat( (embeddings, positions), 2 )

        ## reshape
        ## (batch * 5) x (embed + pos)
        x = x.view(-1, self.embed_dim + 2).float()

        ## fc1
        ## (batch * 5) x hidden1
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc_temp(x) )

        ## reshape
        ## batch x 5 x hidden1
        x = x.view(-1, self.num_objects, self.hidden_dim_1)

        ## get rid of middle object dimension
        ## batch x hidden1
        x = x.max(1)[0].squeeze()

        ## fc2
        ## batch x hidden2
        x = F.relu( self.fc2(x) )

        ## fc3
        ## batch x out
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    inp = torch.LongTensor(5,3).zero_()
    vocab_size = 10
    emb_dim = 3
    rank = 10
    phi = ObjectModel(vocab_size, emb_dim, inp.size(), rank)

    # enc = nn.Embedding(10,emb_dim,padding_idx=0)
    inp = torch.LongTensor(2,5,3).zero_()
    inp[0][0]=1
    inp[0][1]=1
    inp[1][0]=8
    inp[1][1]=8
    print inp
    # inp[0][0][0][0]=1
    # inp[0][1][0][0]=1
    # inp[1][0][0][2]=1
    # print inp
    inp = Variable(inp)

    out = phi.forward(inp)
    # print out
    # out = out.view(-1,2,3,3,emb_dim)
    out = out.data
    print out.size()

    # print out[0][0][0]
    # print out[1][0][0]


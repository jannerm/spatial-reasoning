import sys, math
import numpy as np
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
import pdb, pickle

class ModelFactorizer(nn.Module):
    def __init__(self, state_model, goal_model, state_inp, goal_inp, sparse_value_mat):
        super(ModelFactorizer, self).__init__()
        self.state_model = state_model
        self.goal_model = goal_model
        self.state_inp = Variable(state_inp.cuda())
        self.goal_inp = [Variable(i).cuda() for i in goal_inp]
        self.mat = Variable( sparse_value_mat )
        self.mask = (self.mat == 0)
        self.M = state_inp.size(0)

    def forward(self, inds):
        state_inp = self.state_inp.index_select(0, inds)
        state_out = self.state_model.forward(state_inp)
        goal_out = self.goal_model.forward(self.goal_inp)

        recon = torch.mm(state_out, goal_out.t())
        mask_select = self.mask.index_select(0, inds)
        true_select = self.mat.index_select(0, inds)

        # pdb.set_trace()

        diff = torch.pow(recon - true_select, 2)

        mse = diff.sum()

        return mse

    def train(self, lr, iters, batch_size = 256):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        t = trange(iters)
        for i in t:
            optimizer.zero_grad()
            inds = torch.floor(torch.rand(batch_size) * self.M).long().cuda()
            # bug: floor(rand()) sometimes gives 1
            inds[inds >= self.M] = self.M - 1
            inds = Variable(inds)

            loss = self.forward(inds)
            # print loss.data[0]
            t.set_description( str(loss.data[0]) )
            loss.backward()
            optimizer.step()

        return self.state_model, self.goal_model



if __name__ == '__main__':
    from state_model import *
    from object_model_10 import *
    from text_model import *
    from goal_model import *
    rank = 7
    state_vocab_size = 20
    embed_size = 3
    state_obs_size = (2,10,10)
    goal_obs_size = (1,10,10)
    lstm_size = 15
    lstm_nlayer = 1
    phi = Phi(state_vocab_size, embed_size, state_obs_size, rank).cuda()
    text_model = TextModel(state_vocab_size, lstm_size, lstm_size, lstm_nlayer, lstm_size)
    object_model = ObjectModel(state_vocab_size, embed_size, goal_obs_size, lstm_size)
    psi = Psi(text_model, object_model, lstm_size, lstm_size, rank).cuda()
    print phi

    state_obs = Variable((torch.rand(20*100,2,10,10)*10).long().cuda())
    pdb.set_trace()

    out = phi.forward(state_obs)
    print out.size()
    # from torch.nn.modules.utils import _pair
    # rank = 4
    # # kernel = Variable(avg_conv(rank))
    # kernel = Variable( avg_vector(rank) )

    # inp = Variable(torch.randn(1,rank,20,400))
    # bias = None
    # stride = _pair(1)
    # padding = _pair(1)
    # conv = F.conv2d(inp,kernel,bias,stride,padding)[:,:,1:-1,:]
    # print 'inp:', inp.size()
    # print 'kernel:', kernel.size()
    # print 'conv:', conv.size()

    # pdb.set_trace()

    # from torch.nn.modules.utils import _pair
    # inp = Variable(torch.randn(1,5,8,8))
    # kern = Variable(torch.ones(7,5,3,3))
    # bias = None
    # stride = _pair(1)
    # padding = _pair(1)
    # # dilation = _pair(1)
    # # print dilation
    # # groups = 1
    # # conv = F.conv2d(inp,kern,bias,stride,padding,dilation,groups)
    # conv = F.conv2d(inp,kern,bias,stride,padding)
    # print 'inp:', inp
    # print 'conv: ', conv
    # print 'sum: ', inp[:,:,:3,:3].sum()

    # print a.b
    # print 'conv: ', conv
    value_mat = pickle.load( open('../pickle/value_mat20.p') )
    rank = 10

    print 'value_mat: ', value_mat.shape
    dissimilarity_lambda = .1
    world_lambda = 0
    location_lambda = .001
    model = ConstraintFactorizer(value_mat, rank, dissimilarity_lambda, world_lambda, location_lambda).cuda()
    lr = 0.001
    iters = 500000
    U, V, recon = model.train(lr, iters)

    # lr = 0.001
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    # loss = model.forward( () )
    # print loss
    # iters = 50000
    # t = trange(iters)
    # for i in t:
    #     optimizer.zero_grad()
    #     loss = model.forward( () )
    #     # print loss.data[0]
    #     t.set_description( str(model.mse) + ' ' + str(model.divergence) )
    #     loss.backward()
    #     optimizer.step()

    # U, V = model.embeddings()

    print 'recon'
    print recon
    print 'true'
    print torch.Tensor(value_mat)

    pickle.dump(U, open('../pickle/U_lambda_' + str(dissimilarity_lambda) + '.p', 'w') )
    pickle.dump(V, open('../pickle/V_lambda_' + str(dissimilarity_lambda) + '.p', 'w') )
    # pdb.set_trace()


    # inp = torch.LongTensor(5,3).zero_()
    # vocab_size = 10
    # emb_dim = 3
    # rank = 10
    # phi = ObjectModel(vocab_size, emb_dim, inp.size(), rank)

    # batch = 2
    # hidden = phi.init_hidden(batch)

    # # enc = nn.Embedding(10,emb_dim,padding_idx=0)
    # inp = torch.LongTensor(batch,5,3).zero_()
    # inp[0][0]=1
    # inp[0][1]=1
    # inp[1][0]=8
    # inp[1][1]=8
    # print inp
    # # inp[0][0][0][0]=1
    # # inp[0][1][0][0]=1
    # # inp[1][0][0][2]=1
    # # print inp
    # inp = Variable(inp)

    # out = phi.forward(inp, hidden)
    # # print out
    # # out = out.view(-1,2,3,3,emb_dim)
    # out = out.data
    # print out.size()

    # # print out[0][0][0]
    # # print out[1][0][0]


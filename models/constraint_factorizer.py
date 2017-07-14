import sys, math
import numpy as np
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
import pdb, pickle

class ConstraintFactorizer(nn.Module):
    def __init__(self, sparse_value_mat, rank, dissimilarity_lambda, world_lambda, location_lambda):
        super(ConstraintFactorizer, self).__init__()
        self.M, self.N = sparse_value_mat.shape
        self.mat = Variable( torch.Tensor(sparse_value_mat).cuda() )
        self.mask = (self.mat == 0)
        self.rank = rank
        # print self.M, self.N
        self.M_embed = nn.Embedding(self.M, self.rank)
        self.N_embed = nn.Embedding(self.N, self.rank)
        self.M_inp = Variable( torch.range(0, self.M-1).cuda().long() )
        self.N_inp = Variable( torch.range(0, self.N-1).cuda().long() )
        self.dim = int(math.sqrt(self.N))

        # self.conv_kernel = torch.ones(1,self.rank,3,3)/8./self.rank
        # self.conv_kernel[:,:,1,1] = -1./self.rank
        self.conv_kernel = Variable( self.avg_conv(self.rank).cuda() )
        self.conv_bias = None
        self.stride = _pair(1)
        self.padding = _pair(1)

        ######## similarity loss on states
        # self.conv_vector = self.avg_vector(self.rank)
        ########

        # print 'BEFORE PARAMETER:'
        # print self.conv_kernel
        # pdb.set_trace()
        # self.param = self.conv_kernel 
        # print 'CONV KERNEL:'
        # print self.param
        # self.conv_kernel = self.conv_kernel.cuda()
        # self.conv_bias = self.conv_bias.cuda()
        # self.conv = nn.Conv2d(self.rank,1,kernel_size=3,padding=1)

        self.dissimilarity_lambda = dissimilarity_lambda
        self.world_lambda = world_lambda
        self.location_lambda = location_lambda

        # print self.mat
        # print self.mask

    def __lookup(self):
        rows = self.M_embed(self.M_inp)
        columns = self.N_embed(self.N_inp)
        return rows, columns

    # def __reset_conv(self):
    #     # print self.conv_kernel
    #     self.conv.weight = nn.Parameter(self.conv_kernel)
    #     self.conv.bias = nn.Parameter(self.conv_bias)

    ## batch x 5 x 3
    def forward(self, x):
        # self.__reset_conv()
        # print self.conv.weight
        # print self.conv.bias
        ## batch x 5 x 1
        rows, columns = self.__lookup()
        out = torch.mm(rows, columns.t())
        # print out
        out[self.mask] = 0

        # print out
        # print self.mat
        diff = torch.pow(out - self.mat, 2)
        mse = diff.sum() 

        ## 1 x M x N x rank
        layout = columns.view(self.dim,self.dim,self.rank).unsqueeze(0)
        ## 1 x rank x N x M
        layout = layout.transpose(1,3)
        ## 1 x rank x M x N
        layout = layout.transpose(2,3)

        average = F.conv2d(layout,self.conv_kernel,self.conv_bias,self.stride,self.padding)
        divergence_penalty = torch.pow(layout - average, 2).sum()

        # print 'divergenec: ', divergence_penalty.size()
        # print 'layout: ', layout.size()
        # conv = self.conv(layout)
        # divergence_penalty = conv.sum()
        # print 'conv: ', conv.size()
        # pdb.set_trace()
        self.mse = mse.data[0]
        self.divergence = divergence_penalty.data[0]

        loss = mse + self.dissimilarity_lambda * divergence_penalty

        ######## similarity loss on states
        ## worlds x state_size x rank
        states = rows.view(self.M / self.N, self.N, self.rank)

        world_avg = states.mean(1).repeat(1,self.N,1)
        location_avg = states.mean(0).repeat(self.M/self.N,1,1)
        # pdb.set_trace()
        world_mse = torch.pow(states - world_avg, 2).sum()
        location_mse = torch.pow(states - location_avg, 2).sum()
        self.world_mse = world_mse.data[0]
        self.location_mse = location_mse.data[0]

        loss += (self.world_lambda * world_mse) + (self.location_lambda * location_mse)
        ## rank x state_size x worlds
        # states = states.transpose(0,2)
        ## rank x worlds x state_size
        # states = states.transpose(1,2)
        # F.conv2d(states, self.conv_vector,self.conv_bias,self.stride,self.padding)
        ########


        # print rows.size(), columns.size(), out.size()
        # print out
        # print out
        # print loss

        return loss

    def embeddings(self):
        return self.__lookup()

    def avg_conv(self, out_dim):
        kernel = torch.zeros(out_dim,out_dim,3,3)
        for i in range(out_dim):
            kernel[i][i] = 1./8
            kernel[i][i][1][1] = 0
        return kernel

    # def avg_vector(self, out_dim):
    #     kernel = torch.zeros(out_dim,out_dim,1,3)
    #     for i in range(out_dim):
    #         kernel[i][i][0] = torch.Tensor((1,0,1))/2.
    #     return kernel

    def train(self, lr, iters):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        t = trange(iters)
        for i in t:
            optimizer.zero_grad()
            loss = self.forward( () )
            # print loss.data[0]
            t.set_description( '%.3f | %.3f | %.3f | %.3f' % (self.mse, self.divergence, self.world_mse, self.location_mse) )
            loss.backward()
            optimizer.step()

        U, V = self.__lookup()
        recon = torch.mm(U, V.t())
        # print U, V, recon
        U = U.data.cpu().numpy()
        V = V.data.cpu().numpy()
        recon = recon.data.cpu().numpy()
        return U, V, recon

def avg_conv(out_dim):
    kernel = torch.zeros(out_dim,out_dim,3,3)
    for i in range(out_dim):
        kernel[i][i] = 1./8
        kernel[i][i][1][1] = 0
    return kernel

def avg_vector(out_dim):
    kernel = torch.zeros(out_dim,out_dim,1,3)
    for i in range(out_dim):
        kernel[i][i][0] = torch.Tensor((1,0,1))/2.
    return kernel

if __name__ == '__main__':
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


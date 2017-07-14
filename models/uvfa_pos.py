## predicts entire value map
## rather than a single value

import torch
import math, torch.nn as nn, pdb
import torch.nn.functional as F
from torch.autograd import Variable
import models, utils

class UVFA_pos(nn.Module):
    def __init__(self, state_vocab, object_vocab, args, map_dim = 10, batch_size = 32):
        super(UVFA_pos, self).__init__()
        
        self.state_vocab = state_vocab
        self.object_vocab = object_vocab
        self.total_vocab = state_vocab + object_vocab
        self.pos_size = 2

        self.rank = args.rank
        self.map_dim = map_dim
        self.batch_size = batch_size
        self.positions = self.__agent_pos()

        ## add one for agent position
        self.input_dim = (self.total_vocab + 1) * (map_dim**2)
        self.world_layers = [self.input_dim, 128, 128, args.rank]
        self.world_mlp = models.MLP(self.world_layers)

        # self.object_dim = self.object_vocab * (map_dim**2)
        self.pos_layers = [self.pos_size, 128, 128, args.rank]
        self.pos_mlp = models.MLP(self.pos_layers)

    '''
    returns tensor with one-hot vector encoding
    [1, 2, 3, ..., map_dim] repeated batch_size times
    < batch_size * map_dim, state_vocab >
    '''
    def __agent_pos(self):
        size = self.map_dim**2
        positions = torch.zeros(self.batch_size*size, 100, 1)
        # print positions.size()
        for ind in range(size):
            # print ind, ind*self.batch_size, (ind+1)*self.batch_size, ind, positions.size()
            # positions[ind*self.batch_size:(ind+1)*self.batch_size, ind] = 1
            positions[ind:self.batch_size*size:size, ind] = 1
        # pdb.set_trace()
        return Variable( positions.cuda() )

    def __repeat_position(self, x):
        if x.size() == 2:
            return x.unsqueeze(1).repeat(1,self.map_dim**2,1)
        else:
            return x.unsqueeze(1).repeat(1,self.map_dim**2,1,1)

    '''
    < batch_size x N >
    < batch_size*100 x N >
    '''
    def __construct_inp(self, world, pos):
        world = self.__repeat_position(world)
        world = world.view(self.batch_size*self.map_dim**2,self.map_dim**2,self.total_vocab)
        ## add agent position
        world = torch.cat( (world, self.positions), -1)
        ## reshape to (batched) vector for input to MLPs
        world = world.view(-1, self.input_dim) 
        
        # obj = self.__repeat_position(obj)
        # obj = obj.view(self.batch_size*self.map_dim**2,self.map_dim**2,self.object_vocab)
        # obj = obj.view(-1, self.object_dim) 

        pos = self.__repeat_position(pos)
        # pos = pos.view(self.batch_size*self.map_dim**2,self.map_dim**2,self.pos_size)
        pos = pos.view(-1, self.pos_size) 

        return world, pos


    def forward(self, inp):
        (state, obj, pos) = inp
        batch_size = state.size(0)
        # text = text.transpose(0,1)
        # hidden = self.lstm.init_hidden(batch_size * self.map_dim**2)

        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.positions = self.__agent_pos()

        ## reshape to (batched) vectors
        ## can't scatter Variables
        state = state.data.view(-1, self.map_dim**2, 1)
        obj = obj.data.view(-1, self.map_dim**2, 1)

        ## make state / object indices into one-hot vectors
        state_binary = torch.zeros(batch_size, self.map_dim**2, self.total_vocab).cuda() 
        object_binary = torch.zeros(batch_size, self.map_dim**2, self.total_vocab).cuda() 
        state_binary.scatter_(2, state, 1)
        object_binary.scatter_(2, obj+self.state_vocab, 1)

        ## < batch x 100 x total_vocab >
        ## state_binary will only have non-zero components in the first state_vocab components
        ## object_binary will only have non-zero components in state_vocab:total_vocab components
        input_binary = state_binary + object_binary
        # pdb.set_trace()

        input_binary = Variable( input_binary )
        # object_binary = Variable( object_binary )
        # print input_binary.size(), pos.size()
        # pdb.set_trace()
        input_binary, pos = self.__construct_inp(input_binary, pos)

        # print state_binary.size(), object_binary.size(), text.size()
        # object_binary = self.__repeat_position(object_binary)
        # object_binary = object_binary.view(self.batch_size*self.map_dim**2,self.map_dim**2,self.object_vocab)

        ## add in agent position
        ## < batch x 100 x 2 >
        ## < batch x 100 x 100 x 2 >
        # state_binary = self.__repeat_position(state_binary)
        ## < batch*100 x 100 x 2 >
        # state_binary = state_binary.view(self.batch_size*self.map_dim**2,self.map_dim**2,self.state_vocab)

        ## add agent position
        # state_binary = torch.cat( (state_binary, self.positions), -1)

        # pdb.set_trace()
        # print state_binary.size(), object_binary.size()

        ## reshape to (batched) vectors for input to MLPs
        ## turn back into Variables for backprop
        # state_binary = state_binary.view(-1, self.state_dim) 
        # object_binary = object_binary.view(-1, self.object_dim) 
        # print input_binary.size()
        # pdb.set_trace()
        ## < batch*100 x rank >
        world_out = self.world_mlp(input_binary)
        pos_out = self.pos_mlp(pos)

        # lstm_out = self.lstm.forward(text, hidden)


        # print lstm_out.size()

        # print world_out.size(), pos_out.size()

        values = world_out * pos_out
        map_pred = values.sum(1).view(self.batch_size, self.map_dim, self.map_dim)


        return map_pred


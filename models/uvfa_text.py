## predicts entire value map
## rather than a single value

import torch
import math, torch.nn as nn, pdb
import torch.nn.functional as F
from torch.autograd import Variable
import models, utils

class UVFA_text(nn.Module):
    def __init__(self, lstm, state_vocab, object_vocab, args, map_dim = 10, batch_size = 32):
        super(UVFA_text, self).__init__()
        
        self.state_vocab = state_vocab
        self.object_vocab = object_vocab
        self.lstm = lstm
        self.rank = args.rank
        self.map_dim = map_dim
        self.batch_size = batch_size
        self.positions = self.__agent_pos()

        ## add one for agent position
        self.state_dim = (self.state_vocab+1) * (map_dim**2)
        # self.state_dim = self.state_vocab * map_dim**2
        self.state_layers = [self.state_dim, 128, 128, args.rank]
        self.state_mlp = models.MLP(self.state_layers)

        self.object_dim = self.object_vocab * (map_dim**2)
        self.object_layers = [self.object_dim, 128, 128, args.rank]
        self.object_mlp = models.MLP(self.object_layers)

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
    def __construct_inp(self, state, obj, text):
        state = self.__repeat_position(state)
        state = state.view(self.batch_size*self.map_dim**2,self.map_dim**2,self.state_vocab)
        ## add agent position
        state = torch.cat( (state, self.positions), -1)
        ## reshape to (batched) vector for input to MLPs
        state = state.view(-1, self.state_dim) 
        
        obj = self.__repeat_position(obj)
        obj = obj.view(self.batch_size*self.map_dim**2,self.map_dim**2,self.object_vocab)
        obj = obj.view(-1, self.object_dim) 

        instr_length = text.size(1)
        ## < batch x length >
        ## < batch x 100 x length >
        text = self.__repeat_position(text)
        ## < batch*100 x length >
        text = text.view(self.batch_size*self.map_dim**2,instr_length)
        ## < length x batch*100 >
        text = text.transpose(0,1)
        ## < batch*100 x rank >

        return state, obj, text


    def forward(self, inp):
        (state, obj, text) = inp
        batch_size = state.size(0)
        # text = text.transpose(0,1)
        hidden = self.lstm.init_hidden(batch_size * self.map_dim**2)

        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.positions = self.__agent_pos()

        ## reshape to (batched) vectors
        ## can't scatter Variables
        state = state.data.view(-1, self.map_dim**2, 1)
        obj = obj.data.view(-1, self.map_dim**2, 1)

        ## make state / object indices into one-hot vectors
        state_binary = torch.zeros(batch_size, self.map_dim**2, self.state_vocab).cuda() 
        object_binary = torch.zeros(batch_size, self.map_dim**2, self.object_vocab).cuda() 
        state_binary.scatter_(2, state, 1)
        object_binary.scatter_(2, obj, 1)

        state_binary = Variable( state_binary )
        object_binary = Variable( object_binary )

        state_binary, object_binary, text = self.__construct_inp(state_binary, object_binary, text)

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

        ## < batch*100 x rank >
        state_out = self.state_mlp(state_binary)
        object_out = self.object_mlp(object_binary)

        lstm_out = self.lstm.forward(text, hidden)


        # print lstm_out.size()

        values = state_out * object_out * lstm_out
        map_pred = values.sum(1).view(self.batch_size, self.map_dim, self.map_dim)


        return map_pred

    # def forward(self, inp):
    #     (state, obj, text) = inp
    #     batch_size = state.size(0)
    #     text = text.transpose(0,1)
    #     hidden = self.lstm.init_hidden(batch_size)

    #     if batch_size != self.batch_size:
    #         self.batch_size = batch_size
    #         # self.positions = self.__agent_pos()

    #     ## reshape to (batched) vectors
    #     ## can't scatter Variables
    #     state = state.data.view(-1, self.map_dim**2, 1)
    #     obj = obj.data.view(-1, self.map_dim**2, 1)

    #     ## make state / object indices into one-hot vectors
    #     state_binary = torch.zeros(batch_size, self.map_dim**2, self.state_vocab).cuda() 
    #     object_binary = torch.zeros(batch_size, self.map_dim**2, self.object_vocab).cuda() 
    #     state_binary.scatter_(2, state, 1)
    #     object_binary.scatter_(2, obj, 1)

    #     state_binary = Variable( state_binary )
    #     object_binary = Variable( object_binary )

    #     ## add in agent position
    #     ## < batch x 100 x 2 >
    #     ## < batch x 100 x 100 x 2 >
    #     # state_binary = state_binary.unsqueeze(1).repeat(1,self.map_dim**2,1,1)
    #     # state_binary = self.__repeat_position(state_binary)
    #     ## < batch*100 x 100 x 2 >
    #     # state_binary = state_binary.view(self.batch_size*self.map_dim**2,self.map_dim**2,self.state_vocab)

    #     ## add agent position
    #     # state_binary = torch.cat( (state_binary, self.positions), -1)

    #     # pdb.set_trace()
    #     # print state_binary.size(), object_binary.size()

    #     ## reshape to (batched) vectors for input to MLPs
    #     ## turn back into Variables for backprop
    #     state_binary = state_binary.view(-1, self.state_dim) 
    #     object_binary = object_binary.view(-1, self.object_dim) 
    #     print 'state: ', state_binary.size(), object_binary.size()
    #     state_out = self.state_mlp(state_binary)
    #     object_out = self.object_mlp(object_binary)

    #     # print state_out.size(), object_out.size()

    #     lstm_out = self.lstm.forward(text, hidden)

    #     # object_out = self.__repeat_position(object_out).view(-1, self.rank)
    #     # lstm_out = self.__repeat_position(lstm_out).view(-1, self.rank)

    #     # print state_out.size(), object_out.size(), lstm_out.size()

    #     values = state_out * object_out * lstm_out
    #     # map_pred = values.sum(1).view(self.batch_size, self.map_dim**2)
    #     values = values.sum(1)
    #     # print values.size()
    #     map_pred = values.unsqueeze(-1).repeat(1,self.map_dim,self.map_dim)
    #     print values.size(), map_pred.size()

    #     return map_pred















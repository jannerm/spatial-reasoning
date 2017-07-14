## attention model with only a single convolution
## LSTM kernel isn't really used as attention map,
## but just as a kernel for convolution

import torch
import math, torch.nn as nn, pdb
import torch.nn.functional as F
from torch.autograd import Variable
import models, utils

class CNN_LSTM(nn.Module):
# args.lstm_out, args.goal_hid, args.rank, args.obj_embed
    def __init__(self, state_model, object_model, lstm, args, map_dim = 10, batch_size = 32):
        super(CNN_LSTM, self).__init__()
        
        self.state_model = state_model
        self.object_model = object_model

        self.cnn_inp_dim = args.obj_embed + args.state_embed + 1
        self.cnn = models.ConvToVector(self.cnn_inp_dim)
        self.lstm = lstm

        self.fc1 = nn.Linear(args.cnn_out_dim, 16)
        self.fc2 = nn.Linear(16, 1)

        # self.state_vocab = args.state_embed
        # self.object_vocab = args.obj_embed

        self.state_dim = args.state_embed
        self.object_dim = args.obj_embed

        self.map_dim = map_dim
        self.batch_size = batch_size
        self.positions = self.__agent_pos_2d()

    # '''
    # returns tensor with one-hot vector encoding
    # [1, 2, 3, ..., map_dim] repeated batch_size times
    # < batch_size * map_dim, state_vocab >
    # '''
    # def __agent_pos(self):
    #     size = self.map_dim**2
    #     positions = torch.zeros(self.batch_size*size, 100, 1)
    #     # print positions.size()
    #     for ind in range(size):
    #         # print ind, ind*self.batch_size, (ind+1)*self.batch_size, ind, positions.size()
    #         # positions[ind*self.batch_size:(ind+1)*self.batch_size, ind] = 1
    #         positions[ind:self.batch_size*size:size, ind] = 1
    #     # pdb.set_trace()
    #     return Variable( positions.cuda() )

    def __agent_pos_2d(self):
        ## < 10 x 10 >
        positions = torch.zeros(self.map_dim**2, self.map_dim**2)
        for i in range(self.map_dim**2):
            positions[i][i] = 1

        ## < 100 x 10 x 10 >
        positions = positions.view(self.map_dim**2, self.map_dim, self.map_dim)
        ## < 100 x 1 x 10 x 10 >
        ## < 100*batch x 1 x 10 x 10 >
        positions = positions.unsqueeze(1).repeat(self.batch_size,1,1,1)
        return Variable( positions.cuda() )


    def __repeat_position(self, x):
        # print 'X: ', x.size()
        if x.size() == 2:
            return x.unsqueeze(1).repeat(1,self.map_dim**2,1)
        elif x.size() == 3:
            return x.unsqueeze(1).repeat(1,self.map_dim**2,1,1)
        else:
            return x.repeat(1,self.map_dim**2,1,1)

    # '''
    # < batch_size x N >
    # < batch_size*100 x N >
    # '''
    # def __construct_inp(self, state, obj, text):
    #     state = self.__repeat_position(state)
    #     # pdb.set_trace()
    #     state = state.view(self.batch_size*self.map_dim**2,self.map_dim**2,self.state_dim)
    #     ## add agent position
    #     state = torch.cat( (state, self.positions), -1)
    #     ## reshape to (batched) vector for input to MLPs
    #     state = state.view(-1, self.state_dim+1) 
        
    #     obj = self.__repeat_position(obj)
    #     obj = obj.view(self.batch_size*self.map_dim**2,self.map_dim**2,self.object_dim)
    #     obj = obj.view(-1, self.object_dim) 

    #     instr_length = text.size(1)
    #     ## < batch x length >
    #     ## < batch x 100 x length >
    #     text = self.__repeat_position(text)
    #     ## < batch*100 x length >
    #     text = text.view(self.batch_size*self.map_dim**2,instr_length)
    #     ## < length x batch*100 >
    #     text = text.transpose(0,1)
    #     ## < batch*100 x rank >

    #     return state, obj, text

    def forward(self, inp):
        (state, obj, text) = inp
        batch_size = state.size(0)

        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.positions = self.__agent_pos_2d()

        ## < batch x layout x 10 x 10 >
        state_out = self.state_model(state)
        ## < batch x object x 10 x 10 >
        obj_out = self.object_model.forward(obj)

        ## < batch x layout+object x 10 x 10 >
        embeddings = torch.cat( (state_out, obj_out), 1 )
        ## < batch x (layout+object)*100 x 10 x 10 >
        ## < batch*100 x layout+object x 10 x 10 >
        embeddings = self.__repeat_position(embeddings).view(self.batch_size*self.map_dim**2,self.cnn_inp_dim-1,self.map_dim,self.map_dim)

        ## < batch*100, layout+object+agent, 10, 10 >
        ## state + object embeddings + agent position
        concat = torch.cat( (embeddings, self.positions), 1)
        ## < batch*100 x 1 x 4 x 4 >
        ## < batch*100 x 16 > 
        cnn_out = self.cnn( concat ).view(batch_size*self.map_dim**2, -1)


        instr_length = text.size(1)
        ## < batch x length >
        ## < batch x 100 x length >
        text = self.__repeat_position(text)
        ## < batch*100 x length >
        text = text.view(self.batch_size*self.map_dim**2,instr_length)
        ## < length x batch*100 >
        text = text.transpose(0,1)
        hidden = self.lstm.init_hidden(self.batch_size*self.map_dim**2)
        ## < batch*100 x rank >
        lstm_out = self.lstm.forward(text, hidden)

        concat = torch.cat( (cnn_out, lstm_out), 1 )

        out = F.relu(self.fc1(concat))
        out = self.fc2(out)
        
        map_pred = out.view(self.batch_size,self.map_dim,self.map_dim)

        return map_pred


if __name__ == '__main__':
    from text_model import *
    from object_model import *
    from lookup_model import *

    batch = 2
    seq = 10

    text_vocab = 10
    lstm_inp = 5
    lstm_hid = 3
    lstm_layers = 1

    # obj_inp = torch.LongTensor(1,10,10).zero_()
    obj_vocab = 3
    emb_dim = 3

    concat_dim = 27
    hidden_dim = 5
    out_dim = 7

    text_model = TextModel(text_vocab, lstm_inp, lstm_hid, lstm_layers, concat_dim)
    object_model = LookupModel(obj_vocab, emb_dim, concat_dim)

    psi = AttentionModel(text_model, object_model, concat_dim, hidden_dim, out_dim)

    hidden = text_model.init_hidden(batch)
    text_inp = Variable(torch.floor(torch.rand(batch,seq)*text_vocab).long())
    obj_inp = Variable(torch.floor(torch.rand(batch,1,10,10)*obj_vocab).long())

    to = psi.forward( (obj_inp, text_inp) )
    print to.size()

    # # inp = Variable(torch.LongTensor((1,2,3)))
    # print 'INPUT: ', text_inp
    # print text_inp.size()
    # out = text_model.forward( text_inp, hidden )

    # print 'OUT: ', out
    # print out.size()
    # # print 'HID: ', hid

    # obj_inp = Variable(torch.LongTensor(5,1,20,20).zero_())
    # obj_inp = Variable(obj_inp)
    # obj_out = object_model.forward(obj_inp)

    # print obj_out.data.size()









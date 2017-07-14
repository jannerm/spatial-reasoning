import torch
import math, torch.nn as nn
import torch.nn.functional as F
import utils

class AttentionModel(nn.Module):

    def __init__(self, text_model, object_model, args, final_hidden = 20, map_dim = 10):
        super(AttentionModel, self).__init__()
        
        assert args.attention_kernel % 2 == 1

        self.text_model = text_model
        self.object_model = object_model

        self.embed_dim = args.obj_embed
        self.kernel_out_dim = args.attention_out_dim
        self.kernel_size = args.attention_kernel

        self.conv_custom = utils.ConvKernel(self.embed_dim, self.kernel_out_dim, self.kernel_size, bias=False, padding=1)

        ## final_hidden and fc1 are hard-coded
        ## should be bash arg / inferred
        self.conv1_kernel = 3
        self.conv1 = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=self.conv1_kernel, padding=1)
        # self.conv2 = nn.Conv2d(3, 6, kernel_size=5)
        # self.conv3 = nn.Conv2d(6,12, kernel_size=5)
        self.reshape_dim = self.embed_dim * (map_dim)**2
        self.fc1 = nn.Linear(self.reshape_dim, final_hidden)
        self.fc2 = nn.Linear(final_hidden, args.rank)

    def __conv(self, inp, kernel):
        # print '__conv: ', inp.size(), kernel.size()
        batch_size = inp.size(0)
        out = [ self.conv_custom(inp[i].unsqueeze(0), kernel[i]) for i in range(batch_size) ]
        out = torch.cat(out, 0)
        return out

    def forward(self, inp):
        (obj, text) = inp
        batch_size = obj.size(0)
        text = text.transpose(0,1)
        hidden = self.text_model.init_hidden(batch_size)

        embeddings = self.object_model.forward(obj)

        lstm_out = self.text_model.forward(text, hidden)
        lstm_out = lstm_out.view(-1, 1, self.embed_dim, self.kernel_size, self.kernel_size)
        # print 'lstm: ', lstm_out.size()

        ## first convolve the object embeddings
        ## to populate the states neighboring objects
        ## (otherwise they would be all 0's)
        conv = F.relu(self.conv1(embeddings))
        # print 'conv: ', conv.size()
        ## get attention map
        attention = self.__conv(conv, lstm_out)
        # print 'attention: ', attention.size()
        # attention = self.__conv(embeddings, lstm_out)
        attention = attention.repeat(1,self.embed_dim,1,1)
        # print 'conv: ', conv.size()
        attended = attention * conv
        attended = attended.view(-1, self.reshape_dim)
        # print 'attended: ', attended.size()
        out = F.relu(self.fc1(attended))
        out = F.relu(self.fc2(out))

        # out = F.relu(self.conv1(attended))
        # out = F.relu(self.conv2(out))
        # out = F.relu(self.conv3(out))
        # out = out.view(-1, 12*2*2)
        # out = F.relu(self.fc1(out))
        # out = self.fc2(out)

        return out


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









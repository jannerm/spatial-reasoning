import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

'''
Text inputs are seq x batch
'''
class TextModel(nn.Module):

    def __init__(self, vocab_size, ninp, nhid, nlayers, out_dim):
        super(TextModel, self).__init__()
        
        self.rnn_type = 'LSTM'
        self.nhid = nhid
        self.nlayers = nlayers
        self.out_dim = out_dim

        self.encoder = nn.Embedding(vocab_size, ninp, padding_idx=0)
        self.rnn = nn.LSTM(ninp, nhid, nlayers)
        self.decoder = nn.Linear(nhid, out_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp, hidden):
        emb = self.encoder(inp)
        # print 'emb: ', emb.size()
        output, hidden = self.rnn(emb, hidden)
        # print 'TEXT: ', output.size()
        final_output = output[-1,:,:]
        decoded = self.decoder(final_output)
        return decoded

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))


if __name__ == '__main__':
    batch = 12
    seq = 4
    vocab = 20
    ninp = 5
    nhid = 3
    nlayers = 2
    out_dim = 7
    rnn = TextModel(vocab, ninp, nhid, nlayers, out_dim)

    hidden = rnn.init_hidden(batch)
    ## inds x batch
    ## 
    inp = torch.floor(torch.rand(seq,batch)*vocab).long()
    inp[0,:5] = 0
    inp[1,:5] = 0
    # inp[2,:] = 0
    # inp[3,:] = 0
    inp = Variable(inp)
    # inp = Variable(torch.LongTensor((1,2,3)))
    print 'INPUT: ', inp
    print inp.size()
    out = rnn.forward( inp,hidden )

    print 'OUT: ', out
    print out.size()
    # print 'HID: ', hid









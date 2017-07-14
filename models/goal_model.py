import torch
import torch.nn as nn
import torch.nn.functional as F

class Psi(nn.Module):

    def __init__(self, text_model, object_model, concat_dim, hidden_dim, out_dim):
        super(Psi, self).__init__()
        
        self.text_model = text_model
        self.object_model = object_model
        self.fc1 = nn.Linear(concat_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)


    def forward(self, inp):
        (obj, text) = inp
        batch_size = obj.size(0)
        text = text.transpose(0,1)
        hidden = self.text_model.init_hidden(batch_size)
        # print obj, text, hidden

        obj_out = self.object_model.forward(obj)
        # print text
        # print 'obj out: ', obj_out.data.size()
        # print 'text inp: ', text.data.size()
        # print 'hidden: ', hidden.data.size()
        text_out = self.text_model.forward(text, hidden)
        concat = F.relu( torch.cat((obj_out, text_out), 1) )
        output = F.relu(self.fc1(concat))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


if __name__ == '__main__':
    from text_model import *
    from object_model import *

    batch = 2
    seq = 10

    text_vocab = 10
    lstm_inp = 5
    lstm_hid = 3
    lstm_layers = 2

    obj_inp = torch.LongTensor(1,20,20).zero_()
    obj_vocab = 3
    emb_dim = 3

    concat_dim = 10
    hidden_dim = 5
    out_dim = 7

    text_model = TextModel(text_vocab, lstm_inp, lstm_hid, lstm_layers, concat_dim)
    object_model = ObjectModel(obj_vocab, emb_dim, obj_inp.size(), concat_dim)

    psi = Psi(text_model, object_model, concat_dim, hidden_dim, out_dim)

    hidden = text_model.init_hidden(batch)
    text_inp = Variable(torch.floor(torch.rand(seq,batch)*text_vocab).long())
    obj_inp = Variable(torch.floor(torch.rand(batch,1,20,20)*obj_vocab).long())

    to = psi.forward(obj_inp, text_inp, hidden)
    print to

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









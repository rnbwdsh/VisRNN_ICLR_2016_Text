import torch
import torch.nn as nn
from torch.autograd import Variable

import sys

from .RNN import *
from .LSTM import *

import pdb


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="lstm", n_layers=1):
        super(CharRNN, self).__init__()

        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, output_size)

        if self.model == 'rnn':
            self.rnn = RNN(input_size, hidden_size, n_layers, batch_first=True)
        elif self.model == 'lstm':
            self.rnn = LSTM(input_size, hidden_size, n_layers, batch_first=True)
        elif self.model == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
        else:
            raise Exception('No such a model! Exit.')
            sys.exit(-1)

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, init_hidden):
        encoded = self.encoder(input)  # input: (batch)
        output, hidden, gates = self.rnn(encoded.view(input.shape[0], 1, -1),
                                         init_hidden)  # encoded: (batch, 1, input_size)
        decoded = self.decoder(output)  # output: (batch, 1, hidden_size * num_directions)

        return decoded, hidden, gates  # decoded: (batch, seq_len, output_size)

    def init_hidden(self, batch_size):
        if self.model == 'lstm':
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


class new_CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=612, n_layers=2,
                 drop_prob=0.5, lr=0.001, train_on_gpu=True, model='lstm'):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.model_name = model.lower()

        # creating character dictionaries
        self.int2char, self.char2int = tokens
        self.chars = tuple(self.char2int.keys())

        ## TODO: define the model
        if self.model_name == 'lstm':
            self.model = nn.LSTM(len(self.chars), n_hidden, n_layers,
                                dropout=drop_prob, batch_first=True)
        elif self.model_name == 'rnn':
            self.model = nn.RNN(len(self.chars), n_hidden, n_layers,
                                dropout=drop_prob, batch_first=True)
        elif self.model_name == 'gru':
            self.model = nn.GRU(len(self.chars), n_hidden, n_layers,
                                dropout=drop_prob, batch_first=True)

        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.char2int.keys()))

    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## TODO: Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.model(x, hidden)

        ## TODO: pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)

        ## TODO: put x through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        train_on_gpu = torch.cuda.is_available()
        weight = next(self.parameters()).data

        if self.model_name=='lstm':
            if (train_on_gpu):
                hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                          weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
            else:
                hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                          weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        else:
            hidden = weight.new(self.n_layers, batch_size, self.n_hidden)
        return hidden

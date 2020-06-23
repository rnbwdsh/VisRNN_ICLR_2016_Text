import json

import os.path as path
import numpy as np
from utils import *
from model import *

import pdb


def vis_cell(test_set, int_to_char, char_to_int, config):
    # no trained model, train a new one
    if not path.exists(path.join(config.model_dir, config.model + '_' + str(config.hidden_size) + '_' + str(config.n_layers) + '.pth')):
        raise Exception('No such a trained model! Please train a new model first!')

    # load a trained model
    char_rnn = CharRNNs(tokens=(int_to_char,char_to_int), n_hidden=config.hidden_size, model=config.model, n_layers=config.n_layers)
    char_rnn.load_state_dict(torch.load(path.join(config.model_dir, config.model + '_' + str(config.hidden_size) + '_' + str(config.n_layers) + '.pth')))
    char_rnn.eval()

    # initialize hidden state
    hidden = char_rnn.init_hidden(1)  # here, batch_size = 1
    if torch.cuda.is_available() and config.cuda:
        hidden = tuple([x.cuda() for x in hidden])

    seq = []  # store all test sequences in character form
    cell = []  # 2d array, store all cell state values; each character corresponds to a row; each row is a c_n
    stop_flag = False  # flag to stop
    counter = 0
    total = 0
    for x, y in get_batches(test_set, config.batch_size, config.seq_length):
        if stop_flag:
            break
        counter+=1
        for i in range(config.seq_length):
            seq.extend([int_to_char[xs] for xs in x[i]])
            hidden = char_rnn.init_hidden(1)  # here, batch_size = 1
            if torch.cuda.is_available() and config.cuda:
                hidden = tuple([x.cuda() for x in hidden])
            # One-hot encode our data and make them Torch tensors
            xp = one_hot_encode(x[i], len(char_to_int.keys()))
            xp, y = torch.from_numpy(xp), torch.from_numpy(y)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            if config.model == 'lstm':
                hidden = tuple([each.data for each in hidden])

            inputs, targets = xp, y
            if config.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                hidden = tuple([each.cuda() for each in hidden])

            total+=1
            out, hidden = char_rnn(inputs, hidden)
            (_,c_n) = hidden
            cell.append(c_n.data.cpu().squeeze().numpy())
            # print progress information
            print('Processing [batch: %d, sequence: %3d]...' % (counter, i))
            print(total)
            if total>=10000:
                stop_flag=True
            if stop_flag:
                break

    # write seq and cell into a json file for visualization
    char_cell = {}
    char_cell['cell_size'] = config.hidden_size
    char_cell['seq'] = ''.join(seq)

    # allocate space for cell values
    for j in range(config.n_layers):
        char_cell['cell_layer_' + str(j + 1)] = []

    total_char = len(cell)
    for i in range(total_char):  # for each character (time step)
        for j in range(config.n_layers):  # for each layer
            char_cell['cell_layer_' + str(j + 1)].append(cell[i][j].tolist())

    with open(path.join(config.vis_dir, 'char_cell.json'), 'w') as json_file:
        json.dump(char_cell, json_file)

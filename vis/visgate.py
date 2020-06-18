from model import *
from utils import *

# visualize gate value


def get_saturated(gate, left_thresh, right_thresh):
    """

    :param left_thresh: the threshold from the left
    :param gate: the specified gate
    :param right_thresh: the threshold from the right
    """
    left_s = []  # length = num_layers
    right_s = []
    total_seq_length = gate.shape[1]  # total_seq_length = total character number
    for i in range(gate.shape[0]):  # for each layer
        left_tmp = gate[i] < left_thresh  # gate[i]: (total_seq_length, hidden_size)
        right_tmp = gate[i] > right_thresh
        left_tmp = np.sum(left_tmp, 0) / total_seq_length  # boradcasting
        right_tmp = np.sum(right_tmp, 0) / total_seq_length
        # add to a list
        left_s.append(left_tmp)
        right_s.append(right_tmp)

    return left_s, right_s  # left_s/right_s: (hidden_size)


def get_gates(test_set, vocab, config):
    # no trained model, train a new one
    if not path.exists(path.join(config.model_dir, config.model + '_' + str(config.hidden_size) + '_' + str(config.n_layers) + '.pth')):
        raise Exception('No such a trained model! Please train a new model first!')

    # load a trained model
    char_rnn = CharRNNs(tokens=vocab, n_hidden=config.hidden_size, model=config.model, n_layers=config.n_layers)
    char_rnn.load_state_dict(torch.load(path.join(config.model_dir, config.model + '.pth')))
    char_rnn.eval()
    # ship to gpu if possible
    hidden = char_rnn.init_hidden(1)  # here, batch_size = 1
    # warmup network
    input_gate = np.empty((config.n_layers, config.seq_length, config.hidden_size))
    forget_gate = np.empty((config.n_layers, config.seq_length, config.hidden_size))
    output_gate = np.empty((config.n_layers, config.seq_length, config.hidden_size))
    # view one sequence as a batch
    for x,y in get_batches(test_set,batch_size=config.batch_size,seq_length=config.seq_length):
        x = one_hot_encode(x, len(vocab[1].keys()))
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        for i in range(config.seq_length):  # for every time step in this batch
            # forward pass, we do not care about output
            _, hidden, gates = char_rnn(x[:,i], hidden) # Need to add gates as a variable that can be returned in network.py line  forward
            # store gate value
            for j in range(config.n_layers):  # for each layer
                input_gate[j][i] = gates['input_gate'][j].data.cpu().numpy().squeeze()
                forget_gate[j][i] = gates['forget_gate'][j].data.cpu().numpy().squeeze()
                output_gate[j][i] = gates['output_gate'][j].data.cpu().numpy().squeeze()
    return input_gate, forget_gate, output_gate


def vis_gate(test_set, vocab, config):
    left_thresh = 0.1  # left saturated threshold
    right_thresh = 0.9  # right saturated threshold
    input_gate, forget_gate, output_gate = get_gates(test_set, vocab, config)
    input_left_s, input_right_s = get_saturated(input_gate, left_thresh, right_thresh)
    forget_left_s, forget_right_s = get_saturated(forget_gate, left_thresh, right_thresh)
    output_left_s, output_right_s = get_saturated(output_gate, left_thresh, right_thresh)

    # saturation plot
    plot_gate((input_left_s, input_right_s), (forget_left_s, forget_right_s), (output_left_s, output_right_s))

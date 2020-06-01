import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import *
from utils import *
from config import *

'''
'''


def train(train_data, val_data, vocab, config, clip=5):
    ''' Training a network

        Arguments
        ---------

        :param clip:
        :param config:
        :param vocab:
        :param val_data:
        :param train_data:

    '''
    train_on_gpu = torch.cuda.is_available()
    net = CharRNNs(tokens=vocab, n_hidden=config.hidden_size, model=config.model, n_layers=config.n_layers)
    net.train()
    epochs = config.max_epochs
    batch_size = config.batch_size
    seq_length = config.seq_length
    print_every = config.print_interval
    lr = config.learning_rate

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    data, val_data = train_data, val_data

    if (train_on_gpu):
        net.cuda()

    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        if torch.cuda.is_available() and config.cuda:
            h = tuple([each.cuda() for each in h])
        for x, y in get_batches(data, batch_size, seq_length):

            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history



            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length))
            for each in h:
                each.detach()
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    if config.model=='lstm':
                        val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if train_on_gpu:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length))

                    val_losses.append(val_loss.item())

                net.train()  # reset to train mode after iterating through validation data

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
    torch.save(net.state_dict(), path.join(config.model_dir, config.model + '.pth'))


def pred(test_set, train_set, val_set, int_to_char, char_to_int, config,top_k):
    if not path.exists(path.join(config.model_dir, config.model + '.pth')):
        train(train_set, val_set, (int_to_char,char_to_int), config)

    # load a trained model
    net = CharRNNs(tokens=(int_to_char,char_to_int), n_hidden=config.hidden_size, model=config.model, n_layers=config.n_layers)
    net.load_state_dict(torch.load(path.join(config.model_dir, config.model + '.pth')))
    net.eval()
    batch_size = config.batch_size
    seq_length = config.seq_length
    test_h = net.init_hidden(batch_size)
    train_on_gpu = torch.cuda.is_available()

    for x, y in get_batches(test_set, batch_size, seq_length):
        # One-hot encode our data and make them Torch tensors
        x = one_hot_encode(x, len(char_to_int.keys()))
        x, y = torch.from_numpy(x), torch.from_numpy(y)


        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        test_h = tuple([each.data for each in test_h])

        inputs, targets = x, y
        if train_on_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()

        out, test_h = net(inputs, test_h)
        # get the character probabilities
        # apply softmax to get p probabilities for the likely next character giving x
        p = F.softmax(out, dim=1).data
        if (train_on_gpu):
            p = p.cpu()  # move to cpu

        # get top characters
        # considering the k most probable characters with topk method
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        # select the likely next character with some element of randomness
        chars = []
        for i in range(p.shape[0]):
            pk = p[i].squeeze()
            pk = pk.numpy()
            char = np.random.choice(top_ch[i], p=pk / pk.sum())
            chars.append(char)
        chars = np.array(chars)
        accuracy = float(np.where(chars == targets.flatten().numpy())[0].size) / chars.size
        print('Overall Accuracy:',accuracy)







if __name__ == '__main__':
    config = get_config()  # get configuration parameters

    # train_set: (input_set, target_set); input_set: (nbatches, batch_size, seq_length)
    #train_set, val_set, test_set, (char_to_int, int_to_char) = create_dataset(config)  # val_set and test_set are similar to train_set

    # train(train_set, val_set, len(char_to_int), config)
    train_set, val_set, test_set, (char_to_int, int_to_char) = create_datasets(config)  # val_set and test_set are similar to train_set


    pred(test_set, train_set, val_set, int_to_char, char_to_int, config,2)

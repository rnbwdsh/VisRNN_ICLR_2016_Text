import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import *
from utils import *
from config import *

import pdb


# build and train a model
def train(train_set, val_set, vocab_size, config):
    # initialize a model
    char_rnn = CharRNN(input_size=len(train_set), hidden_size=config.hidden_size, output_size=vocab_size,
                       model=config.model, n_layers=config.n_layers)
    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        char_rnn.cuda()

    # train_input_set : (train_batches, batch_size, seq_length); the rest are similar
    train_input_set, train_target_set = train_set[0:len(train_set) * .1], train_set[len(train_set) * .1:]
    val_input_set, val_target_set = val_set[0], val_set[1]

    criterion = nn.CrossEntropyLoss()  # include softmax
    optimizer = torch.optim.Adam(char_rnn.parameters(), lr=config.learning_rate)
    # learning rate decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    best_val_loss = sys.float_info.max
    try:
        for epoch_idx in tqdm(range(1, config.max_epochs + 1)):  # for every epoch
            print("Training for %d epochs..." % epoch_idx)
            running_loss = 0.0

            # initialize hidden states for every epoch
            hidden = char_rnn.init_hidden(config.batch_size)  # (n_layers * n_directions, batch_size, hidden_size)
            # ship to gpu if possible
            if torch.cuda.is_available() and config.cuda:
                hidden = tuple([x.cuda() for x in hidden])

            for batch_idx in range(1, train_input_set.shape[0] + 1):  # for every batch
                # for every batch
                char_rnn.zero_grad()  # zero the parameter gradients

                train_input = train_input_set[batch_idx - 1]

                # ship to gpu if possible
                if torch.cuda.is_available() and config.cuda:
                    train_input = train_input.cuda()

                # compute loss for this batch
                loss = 0.0
                for i in range(config.seq_length):  # for every time step in this batch
                    # forward pass
                    train_output, hidden, _ = char_rnn(Variable(train_input[:, i]), hidden)  # ignore gate values
                    # add up loss at each time step
                    loss += criterion(train_output.view(config.batch_size, -1),
                                      train_target_set[batch_idx - 1][:, i])

                # detach hidden state from current computational graph for back-prop
                for x in hidden:
                    x.detach_()

                # backward
                loss.backward()
                optimizer.step()
                scheduler.step()  # lr decay

                # print statistics
                running_loss += loss.item()
                if batch_idx % config.print_interval == 0:  # print_interval batches
                    print('[%d, %4d] loss: %.3f' % (epoch_idx, batch_idx, running_loss / config.print_interval))
                    running_loss = 0.0

                    '''# validate model
                    val_loss = 0
                    # for every batch
                    for val_batch_idx in range(1, val_input_set.shape[0] + 1):

                        val_input = val_input_set[val_batch_idx - 1]

                        # ship to gpu if possible
                        if torch.cuda.is_available() and config.cuda:
                            val_input = val_input.cuda()

                        for i in range(config.seq_length):  # for every time step in this batch
                            # forward pass
                            val_output, _, _ = char_rnn(Variable(val_input[:, i]), hidden)
                            # add up loss at each time step
                            val_loss += criterion(val_output.view(config.batch_size, -1).cpu(),
                                                  Variable(val_target_set[val_batch_idx - 1][:, i]))

                    val_loss /= val_input_set.shape[0]  # loss per batch
                    print('Validation loss: %.3f' % val_loss)

                    # save the best model sofar
                    if val_loss.item() < best_val_loss:
                        print('Saving model [%d, %4d]...' % (epoch_idx, batch_idx))
                        torch.save(char_rnn.state_dict(), path.join(config.model_dir, config.model + '.pth'))
                        # to load a saved model: char_rnn = CharRNN(*args, **kwargs), char_rnn.load_state_dict(
                        # torch.load(PATH))
                        best_val_loss = val_loss.item()'''
        torch.save(char_rnn.state_dict(), path.join(config.model_dir, config.model + '.pth'))

    except KeyboardInterrupt:
        print("Saving before abnormal quit...")
        torch.save(char_rnn.state_dict(), path.join(config.model_dir, config.model + '.pth'))


# use the trained model to make prediction
def pred(test_set, train_set, val_set, int_to_char, vocab_size, config):
    # no trained model, train a new one
    if not path.exists(path.join(config.model_dir, config.model + '.pth')):
        train(train_set, val_set, vocab_size, config)

    # load a trained model
    char_rnn = CharRNN(vocab_size, config.hidden_size, vocab_size, model=config.model, n_layers=config.n_layers)
    char_rnn.load_state_dict(torch.load(path.join(config.model_dir, config.model + '.pth')))
    char_rnn.eval()

    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        char_rnn.cuda()

    # prepare test data
    test_input_set, test_target_set = test_set[0], test_set[1]

    # randomly choose a sequence in train_set to warm up the network
    train_input_set, _ = train_set[0], train_set[1]  # train_input_set: (train_batches, batch_size, seq_length)
    # random batch index
    train_batch_idx = np.random.choice(train_input_set.shape[0])
    # random sequence index
    train_seq_idx = np.random.choice(config.batch_size)
    # random sequence
    warmup_seq = train_input_set[train_batch_idx][train_seq_idx].unsqueeze(0)

    # initialize hidden state
    hidden = char_rnn.init_hidden(1)  # here, batch_size = 1

    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        warmup_seq = warmup_seq.cuda()
        hidden = tuple([x.cuda() for x in hidden])

    # warmup network
    for i in range(config.seq_length):
        # get final hidden state
        _, hidden, _ = char_rnn(Variable(warmup_seq[:, i]), hidden)

    for test_batch_idx in range(1, test_input_set.shape[0] + 1):
        # for every batch
        test_batch = test_input_set[test_batch_idx - 1]
        # for every sequence in this batch
        for test_seq_idx in range(1, config.batch_size + 1):
            # predicted int result (not character)
            pred = []

            # current sequence
            test_seq = test_batch[test_seq_idx - 1]

            # first character in current sequence
            idx = torch.LongTensor([test_seq[0]]).view(1, -1)

            # ship to gpu if possible
            if torch.cuda.is_available() and config.cuda:
                idx = idx.cuda()

            # forward pass
            output, hidden, _ = char_rnn(Variable(idx), hidden)  # idx: (1, 1, input_size); ignore gate values

            # choose the one with the highest value
            prob, idx = torch.topk(output.data, 1, dim=2)
            # add predicted value
            pred.append(idx.cpu().squeeze().item())

            # predict every remaining character in this sequence
            for i in range(1, len(test_seq)):
                # ship to gpu if possible
                if torch.cuda.is_available() and config.cuda:
                    idx = idx.cuda()

                # forward pass
                output, hidden, _ = char_rnn(Variable(idx.view(1, -1)), hidden)  # ingore gate values

                # choose the one with the highest value
                prob, idx = torch.topk(output.data, 1, dim=2)
                # add predicted value
                pred.append(idx.cpu().squeeze().item())

            # calculate prediction accuracy
            pred = np.array(pred)
            target = test_target_set[test_batch_idx - 1][test_seq_idx - 1].numpy()

            accuracy = float(np.where(pred == target)[0].size) / pred.size
            print('Accuracy of [batch %2d, seq %2d] is ' % (test_batch_idx, test_seq_idx) + '{:.2%}'.format(accuracy))

            # convert target and pred from int to character
            target_text = []
            pred_text = []
            for i in range(len(target)):
                target_text.append(int_to_char[target[
                    i]])  # produces error as we get a value == 111 which is not possible because the largest key value is 84 ( will check his int_to_char function
                pred_text.append(int_to_char[pred[i]])

            # display target text and predicted text
            print('Target ----------------------------------------------------------')
            print(''.join(target_text))  # convert from array to string
            print('Predicted -------------------------------------------------------')
            print(''.join(pred_text))

def new_train(train_data, val_data, vocab, config, clip=5):
    ''' Training a network

        Arguments
        ---------

        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    '''
    train_on_gpu = torch.cuda.is_available()
    net = new_CharRNN(tokens=vocab, n_hidden=config.hidden_size, model=config.model, n_layers=config.n_layers)
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

        for x, y in get_batches(data, batch_size, seq_length):

            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if (train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length))
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


def new_pred(test_set, train_set, val_set, int_to_char, char_to_int, config,top_k):
    if not path.exists(path.join(config.model_dir, config.model + '.pth')):
        new_train(train_set, val_set, (int_to_char,char_to_int), config)

    # load a trained model
    net = new_CharRNN(tokens=(int_to_char,char_to_int), n_hidden=config.hidden_size, model=config.model, n_layers=config.n_layers)
    net.load_state_dict(torch.load(path.join(config.model_dir, config.model + '.pth')))
    net.eval()
    batch_size = config.batch_size
    seq_length = config.seq_length
    test_h = net.init_hidden(batch_size)
    train_on_gpu = torch.cuda.is_available()
    correct=0

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
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())
        correct += int(char.eq(targets.view_as(char)).sum().item())

        target_text = []
        pred_text = []
        for i in range(len(targets)):
            target_text.append(int_to_char[test_set[i]])  # produces error as we get a value == 111 which is not possible because the largest key value is 84 ( will check his int_to_char function
            pred_text.append(int_to_char[pred[i]])

        # display target text and predicted text
        print('Accuracy:', correct/len(test_set))
        print('Target ----------------------------------------------------------')
        print(''.join(target_text))  # convert from array to string
        print('Predicted -------------------------------------------------------')
        print(''.join(pred_text))






if __name__ == '__main__':
    config = get_config()  # get configuration parameters

    # train_set: (input_set, target_set); input_set: (nbatches, batch_size, seq_length)
    train_set, val_set, test_set, (char_to_int, int_to_char) = create_dataset(
        config)  # val_set and test_set are similar to train_set

    # train(train_set, val_set, len(char_to_int), config)

    #pred(test_set, train_set, val_set, int_to_char, len(char_to_int), config)
    new_pred(test_set, train_set, val_set, int_to_char, char_to_int, config,5)

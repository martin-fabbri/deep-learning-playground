import time
from collections import namedtuple

import numpy as np
import tensorflow as tf

# load text file and convert it into integers
with open('anna.txt', 'r') as f:
    text = f.read()

vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

print(text[:100])
print(encoded[:100])

print(len(vocab))

def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: (N) Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # get the number of characters per batch and number of batches we can make
    characters_per_batch = n_seqs * n_steps
    n_batches = len(arr) // characters_per_batch

    # keep only enough characters to make full batches
    arr = arr[:n_batches * characters_per_batch]

    # reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        # the features
        x = arr[:, n: n + n_steps]

        # the targets, shifted by one
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


batches = get_batches(encoded, 10, 50)
x, y = next(batches)

print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])

def build_inputs(batch_size, num_steps):
    ''' Define placeholders for inputs, targets, and dropout

        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch

    '''
    # declare placeholders well feed into the graph
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')

    # keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.

        Arguments
        ---------
        keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        batch_size: Batch size

    '''
    ### build the lstm cell
    # use a basic lstm cell
    def build_cell(lstm_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # add dropout to the cell outputs
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    cell = tf.contrib.rnn.MultipleRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state


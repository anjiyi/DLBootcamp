import tensorflow as tf
import math
import numpy as np
import random


from six.moves import xrange  # pylint: disable=redefined-builtin

class config(object):
    
    batch_size = 10
    input_size = 8
    time_step_size = 6
    rnn_hidden = 4
    source_vocab_size = 20
    initializer = tf.contrib.layers.xavier_initializer()

    initial_state = (tf.zeros([batch_size, rnn_hidden]), tf.zeros([batch_size, rnn_hidden]))

def add_placeholder(self):
    
    self.input_placeholder = tf.placeholder(tf.int32, [self.config.batch_size, self.config.input_size])

def add_cell(self):
    self.cell = tf.contrib.rnn.BasicLSTMCell(self.config.rnn_hidden)


def embedding(self):
    """
    input:
    inputs: input placeholder in shape [batch_size, input_size]

    output:
    embedded: embedded input in shape[batch_size*rnn_hidden, time_step_size]
    """
    with tf.device('/cpu:0'):
        ### YOUR CODE HERE
        with tf.variable_scope("embedding") as scope:
            L = tf.get_variable("L",[self.config.source_vocab_size, self.config.rnn_hidden], initializer = self.config.initializer)
            embeds = tf.nn.embedding_lookup(L, self.input_placeholder)
            embedded = [tf.squeeze(x) for x in tf.split(embeds, [tf.ones([self.config.time_step_size], tf.int32)], axis=1)]

            ### END YOUR CODE
    return embedded



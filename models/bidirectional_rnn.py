import models.config as config
import tensorflow as tf
from tensorflow.contrib import rnn


class BIRNNConfig(config.base):
  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_layer=2,
               keep_prob=0.7,
               initializer_range=0.02, *argc, **kwds):
    super(BIRNNConfig, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_layer = num_layer
    self.initializer_range = initializer_range
    self.keep_prob = keep_prob


def create(input_ids,
           config):
  with tf.variable_scope(config.name):
    embedding_table = tf.get_variable(
      name='word_embeddings',
      shape=[config.vocab_size, config.hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=config.initializer_range)
    )

    flat_input_ids = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=config.vocab_size)
    inputs = tf.matmul(one_hot_input_ids, embedding_table)
    input_shape = tf.shape(input_ids)
    inputs = tf.reshape(inputs, [input_shape[0], input_shape[1], config.embedding_size])

    def rnn_unit(keep_prob, scope=None):
      with tf.variable_scope(scope):
        def lstm_cell(keep_prob):
          cell = rnn.LSTMCell(config.hidden_size, forget_bias=1.0, state_is_tuple=True)
          cell = rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
          return cell
      lstm = rnn.MultiRNNCell([lstm_cell(keep_prob) for _ in range(config.num_layer)], state_is_tuple=True)
      return lstm

    forword = rnn_unit(config.keep_prob, 'forword')
    backword = rnn_unit(config.keep_prob, 'backword')
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(forword, backword, inputs, dtype=tf.float32)
    outputs = tf.concat(outputs, axis=-1)
    return outputs

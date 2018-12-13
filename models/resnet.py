import tensorflow as tf
import models.config as config
import tensorflow.contrib.slim as slim
import models.utils as utils
from tensorflow.python.ops import init_ops


class ResnetConfig(config.Base):
  def __init__(self, filters=768, kernel_size=3, block_type='basic',
               num_layers=6, initializer_range=0.02, activation='relu'):
    self.initializer_range = initializer_range
    self.activation = activation
    self.filters = filters
    self.kernel_size = kernel_size
    self.num_layers = num_layers
    self.block_type = block_type


def _basic_block(inputs, filters, kernel_size,
                 dilation_rate=1,
                 activation=tf.nn.relu,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 scope=None,
                 reuse=None):
  with tf.variable_scope(scope, default_name='block'):
    with slim.arg_scope([tf.layers.conv1d],
                        filters=filters,
                        kernel_size=kernel_size,
                        dilation_rate=dilation_rate,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        trainable=trainable,
                        reuse=reuse):
      outputs = tf.layers.conv1d(inputs, name='conv1', activation=activation)
      outputs = tf.layers.batch_normalization(outputs, name='batch_normal1')
      outputs = tf.layers.conv1d(outputs, name='conv2')
      outputs = tf.layers.batch_normalization(outputs, name='batch_normal2')
      outputs = activation(outputs + inputs)
    return outputs


def _bottleneck_block(inputs, filters, kernel_size,
                      dilation_rate=1,
                      activation=tf.nn.relu,
                      use_bias=True,
                      kernel_initializer=None,
                      bias_initializer=init_ops.zeros_initializer(),
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None,
                      bias_constraint=None,
                      trainable=True,
                      scope=None,
                      reuse=None):
  with tf.variable_scope(scope, default_name='block'):
    with slim.arg_scope([tf.layers.conv1d],
                        filters=filters,
                        dilation_rate=dilation_rate,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        trainable=trainable,
                        reuse=reuse):
      outputs = tf.layers.conv1d(inputs, kernel_size=1, name='conv1', activation=activation)
      outputs = tf.layers.batch_normalization(outputs, name='batch_normal1')
      outputs = tf.layers.conv1d(outputs, kernel_size=kernel_size, name='conv2')
      outputs = tf.layers.batch_normalization(outputs, name='batch_normal2')
      outputs = tf.layers.conv1d(outputs, kernel_size=1, name='conv3')
      outputs = tf.layers.batch_normalization(outputs, name='batch_normal2')
      outputs = activation(outputs + inputs)
    return outputs


def resnet_block(blok_type):
  if blok_type == "basic":
    return _basic_block
  elif blok_type == "bottleneck":
    return _bottleneck_block
  else:
    raise ValueError("Unsupported resnet block: %s" % blok_type)


def resnet(inputs, config, scope=None):
  outputs = inputs
  with tf.variable_scope(scope, default_name='resnet'):
    for i in range(config.num_layers):
      inputs = resnet_block(config.block_type)(outputs, config.filters, config.kernel_size,
                                               activation=utils.get_activation(utils.activation),
                                               kernel_initializer=utils.create_initializer(
                                                 config.initializer_range),
                                               scope='layer_%d' % i)
      outputs = inputs
  return outputs

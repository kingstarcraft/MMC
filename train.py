import os
import tensorflow as tf
import pickle
import models.config as config
import bert_resnet
import dataset
import optimization

flags = tf.flags
flags.DEFINE_string('train_dir', './cache/data/train', 'Directory to train dataset.')
flags.DEFINE_string('vocab_path', './Cache/vocab.pkl', 'Path to vocabulary.')
flags.DEFINE_integer('sequence_length', 512, 'Length of squence to clip.')
flags.DEFINE_integer('max_distance', 320, 'Max distance between entity.')
flags.DEFINE_integer('batch_size', 32, 'batch_size used to train.')
flags.DEFINE_integer('neg_pos_ratio', 2, 'the ratio of pos to neg in one batch of train.')
flags.DEFINE_integer('divalte', 2, 'the ratio of pos to neg in one batch of train.')
flags.DEFINE_float('learning_rate''', 5e-5, 'The initial learning rate for Adam.')
flags.DEFINE_integer('max_train_step', 100000, 'The max training steps.')
flags.DEFINE_integer('num_warmup_steps', 10000, 'The interval step of learning rate warmup.')
flags.DEFINE_integer('interval', 500, 'The interval to save model.')
flags.DEFINE_string('bert_config_path', 'bert_config.json', 'Path to config.')
flags.DEFINE_string('resnet_config_path', 'resnet_config.json', 'Path to config.')
flags.DEFINE_string('ckpt_dir', './cache/ckpt', 'Directory to sabe path.')
flags.DEFINE_string('pretrain_dir', './pretrain', 'Path to pretrain.')
flags.DEFINE_integer('num_gpu', 1, 'Num gpu used.')
FLAGS = flags.FLAGS


def average_gradients(gradients, clip_norm=1.0):
  outputs = []
  for grad_and_vars in zip(*gradients):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][2]
    if clip_norm is not None:
      grads, _ = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    grad_and_var = (grad, v)
    outputs.append(grad_and_var)
  return outputs


def main(_):
  vocab = pickle.load(open(FLAGS.vocab_path, 'rb'))
  bert_config = None
  if os.path.exists(FLAGS.config_path):
    bert_config = config.Base.from_json_file(FLAGS.bert_config_path)
  resnet_config = None
  if os.path.exists(FLAGS.config_path):
    resnet_config = config.Base.from_json_file(FLAGS.resnet_config_path)
  train_datasets = dataset.Dataset(FLAGS.train_dir, FLAGS.max_distance, FLAGS.sequence_length)
  train_datasets.Start()

  with tf.device("/cpu:0"):
    inputs = tf.placeholder(shape=[None, 4, FLAGS.sequence_length], dtype=tf.int32)
    labels = tf.placeholder(shape=[None], dtype=tf.int32)
    is_training = tf.placeholder(shape=[], dtype=tf.bool)

    global_step = tf.train.get_or_create_global_step()
    losses = []
    gradients = []

    optimizer = optimization.create_optimizer(FLAGS.learning_rate,
                                              FLAGS.max_train_step,
                                              FLAGS.num_warmup_steps,
                                              global_step)
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(FLAGS.num_gpu):
        with tf.device("/gpu:%d" % i):
          with tf.name_scope("gpu_%d" % i):
            subset_inputs = inputs[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
            subset_labels = labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
            subset_logits = bert_resnet.bulid_bert_resnet_model(vocab, subset_inputs, is_training,
                                                                bert_config=bert_config, resnet_config=resnet_config)
            subset_loss = tf.reduce_mean(
              tf.nn.sparse_softmax_cross_entropy_with_logits(labels=subset_labels, logits=subset_logits))
            gradients.append(optimizer.compute_gradients(subset_loss))
            losses.append(subset_loss)
            tf.get_variable_scope().reuse_variables()
    losses = tf.stack(axis=0, values=losses)
    loss = tf.reduce_mean(losses)
    gradients = average_gradients(gradients)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.apply_gradients(gradients, global_step=global_step)
    saver = tf.train.Saver()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

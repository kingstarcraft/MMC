import tensorflow as tf


def f1_measure(outputs, labels):
  outputs = tf.cast(outputs, labels.dtype)
  predict_postive_mask = tf.greater(outputs, 0, name='predict_postive_mask')
  postives_mask = tf.greater(labels, 0, name='postive_mask')
  ture_positive_mask = tf.equal(
    tf.where(outputs <= 0, tf.zeros_like(outputs) - 1, outputs), labels, name='true_postive_mask')
  num_tp = tf.reduce_sum(tf.cast(ture_positive_mask, dtype=tf.float32), name='num_true_postive')
  num_pred = tf.reduce_sum(tf.cast(predict_postive_mask, dtype=tf.float32), name='num_postive_predict')
  num_pos = tf.reduce_sum(tf.cast(postives_mask, dtype=tf.float32), name='num_pos')
  f1 = tf.div(2 * num_tp + 1e-10, (num_pred + num_pos + 1e-10), name='f1')
  return num_tp, num_pred, num_pos, f1


def accuracy(outputs, labels):
  return tf.reduce_mean(tf.cast(tf.equal(
    tf.cast(tf.reshape(outputs, [-1]), labels.dtype),
    tf.reshape(labels, [-1])),
    dtype=tf.float32))

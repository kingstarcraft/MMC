import pickle
import dataset
import glog
import numpy as np
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string('input_dir', '../cache/data/train', 'Directory to train dataset.')
flags.DEFINE_string('vocab_path', '../cache/vocab.pkl', 'Path to vocabulary.')
FLAGS = flags.FLAGS

if __name__ == '__main__':
  vocab = pickle.load(open(FLAGS.vocab_path, 'rb'))
  test_dataset = dataset.Dataset(FLAGS.input_dir)
  test_dataset.Start()
  batch_size = 20


  def _str_sample(x):
    position = np.where(x[3] != 0)
    words = []
    tokens = []
    token_positions = []
    tabs = ''.join(['\t'] * 13)+'  '
    for p in position[0]:
      words.append(vocab['Word'][x[0][p]])
      tokens.append(x[1][p])
      token_positions.append(x[2][p])
    return "%s%s\n%s%s\n%s%s\n" % (tabs, words, tabs, tokens, tabs, token_positions)


  for j in range(3):
    glog.info("=======================================")
    pos_xs, pos_ys, neg_xs, neg_ys = test_dataset.Get(batch_size)
    for i, pos in enumerate(pos_xs):
      relation = vocab['Relation'][pos_ys[i]]
      info = 'No %d: %s\n%s' % (i, relation, _str_sample(pos))
      glog.info(info)
    for i, neg in enumerate(neg_xs):
      relation = vocab['Relation'][neg_ys[i]]
      info = 'No %d: %s\n%s' % (i, 'negative', _str_sample(neg))
      glog.info(info)
    glog.info("=======================================")
  test_dataset.Stop()
  test_dataset.join()

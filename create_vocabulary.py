import os
import brat
import glog
import pickle
import tensorflow as tf
import collections
import six

flags = tf.flags
flags.DEFINE_string("train_dir", './datasets/ruijin_round2_train/ruijin_round2_train',
                    'Directory to train dataset.')
flags.DEFINE_string("bert_vocab", './chinese_L-12_H-768_A-12/vocab.txt',
                    'Optinal, path to pretrain vocab.')
flags.DEFINE_string("output_path", './cache/vocab.pkl', 'Output path of vocabulary .')
FLAGS = flags.FLAGS


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  stem, ext = os.path.splitext(vocab_file)
  if ext == '.txt':
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
      while True:
        token = convert_to_unicode(reader.readline())
        if not token:
          break
        token = token.strip()
        vocab[token] = index
        index += 1
  elif ext == '.pkl':
    import pickle
    with tf.gfile.GFile(vocab_file, "rb") as reader:
      words = pickle.load(reader)['Word']
      for index, word in enumerate(words):
        vocab[word] = index
  return vocab


def main(_):
  output_dir = os.path.split(FLAGS.output_path)[0]
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
  for i in range(10):
    words.append(str(i))
  for i in range(ord('A'), ord('Z')):
    words.append(chr(i))
  for i in range(ord('a'), ord('z')):
    words.append(chr(i))
  entity = [0]
  relation = [0]

  def process(dir):
    max_length = 0
    for filename in os.listdir(dir):
      filepath = os.path.join(dir, filename)
      stem, ext = os.path.splitext(filename)
      if ext == '.txt':
        datas = brat.Read(filepath)
        for word in datas:
          if word not in words:
            words.append(word)
        max_length = max(max_length, len(datas))
      elif ext == '.ann':
        datas = brat.Read(filepath)
        for id in datas['T']:
          if datas['T'][id][0] not in entity:
            entity.append(datas['T'][id][0])
        for id in datas['R']:
          if datas['R'][id][0] not in relation:
            relation.append(datas['R'][id][0])
      glog.info('Processed %s' % filename)
    return max_length

  max_train_length = process(FLAGS.train_dir)
  # max_test_length = process(FLAGS.test_dir)
  if os.path.exists(FLAGS.bert_vocab):
    vocabs = load_vocab(FLAGS.bert_vocab)
    words = list(vocabs.keys())
  glog.info('Words: %d, Entity: %d, Relation: %d.' % (len(words), len(entity), len(relation)))
  glog.info('Max txt length: %d.' % max_train_length)
  # glog.info('Max txt length: %d.' % max(max_train_length, max_test_length))
  pickle.dump({'Word': words, 'Entity': entity, 'Relation': relation}, open(FLAGS.output_path, 'wb'))


if __name__ == '__main__':
  tf.app.run()

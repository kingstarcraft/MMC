import pickle
import dataset
import numpy as np
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string('input_dir', '../cache/data/train', 'Directory to train dataset.')
flags.DEFINE_string('vocab_path', '../cache/vocab.pkl', 'Path to vocabulary.')
FLAGS = flags.FLAGS

if __name__ == '__main__':
  vocab = pickle.load(open(FLAGS.vocab_path, 'rb'))
  test_dataset = dataset.TrainDataset(FLAGS.input_dir, 32)
  test_dataset.Start()


  def score_func(inputs):
    return [0] * len(inputs)


  for j in range(3):
    xs, ys = test_dataset.Get(2, 3, score_func)
    for i in range(2 + 3):
      x = xs[i]
      y = ys[i]
      relation = vocab['Relation'][y]
      position = np.where(x[3] != 0)
      words = []
      tokens = []
      token_positions = []
      for p in position[0]:
        words.append(vocab['Word'][x[0][p]])
        tokens.append(x[1][p])
        token_positions.append(x[2][p])
      print('batch %d: %s' % (i, relation))
      print(words)
      print(tokens)
      print(token_positions)
    print('\n')
  test_dataset.Stop()
  test_dataset.join()

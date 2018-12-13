import os
import glog
import brat
import relation
import numpy as np
import pickle
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string('input_dir', './datasets/ruijin_round2_train/ruijin_round2_train',
                    'Directory to train dataset.')
flags.DEFINE_integer('train_test_ratio', 10, 'The ratio of train to test.')
flags.DEFINE_string('vocab_path', './cache/vocab.pkl', 'Path to vocabulary, generated from vocabulary.')
flags.DEFINE_string('list_path', './cache/data/list.pkl',
                   'Path to save train or test filename list. if not exit, it will be created.')
flags.DEFINE_string('train_output_dir', './cache/data/train', 'Directory of train dataset to output.')
flags.DEFINE_string('test_output_dir', './cache/data/test', 'Directory of test dataset to output.')
FLAGS = flags.FLAGS


def _create_mmc_dataset(vocab, input_dir, filestems, output_path):
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  max_numbel = 0

  for filestem in filestems:
    words = brat.Read(input_dir + '/' + filestem + '.txt')
    annos = brat.Read(input_dir + '/' + filestem + '.ann')
    encode_words, encode_positions, encode_entities, encode_relations = brat.Encode(words, annos, vocab)
    pred_boxes = relation.create(annos['T'], True)
    encode_relations = list(encode_relations.values())
    ious = relation.ious(pred_boxes, encode_relations)
    pred_ids = np.argmax(ious, axis=1)
    pred_ious = np.max(ious, axis=1)

    matcher = []
    for i in range(len(pred_boxes)):
      iou = pred_ious[i]
      pred = pred_boxes[i]
      if iou >= 0.5:
        ann = encode_relations[pred_ids[i]]
        pred['label'] = ann['label']
        matcher.append(pred)
      else:
        pred['label'] = 0
        matcher.append(pred)

    boxes = matcher
    pickle.dump({'words': encode_words,
                 'positions': encode_positions,
                 'entities': encode_entities,
                 'boxes': boxes},
                open('%s/%s.pkl' % (output_path, filestem), 'wb'))
    glog.info('Processed %s.' % filestem)

    max_numbel = max(len(annos['T']), max_numbel)
  return max_numbel


def main(_):
  vocab = pickle.load(open(FLAGS.vocab_path, 'rb'))
  if not os.path.exists(os.path.split(FLAGS.list_path)[0]):
    os.makedirs(os.path.split(FLAGS.list_path)[0])

  train_list = []
  test_list = []
  if os.path.exists(FLAGS.list_path):
    lists = pickle.load(open(FLAGS.list_path, 'rb'))
    train_list = lists['train']
    test_list = lists['test']
    glog.info('Load list: train:test=%d:%d.' % (len(train_list), len(test_list)))
  else:
    import random
    lists = os.listdir(FLAGS.input_dir)
    random.shuffle(lists)
    count = 0
    for filename in lists:
      stem, ext = os.path.splitext(filename)
      if ext == '.txt':
        annpath = FLAGS.input_dir + '/' + stem + '.ann'
        if os.path.exists(annpath):
          if count % FLAGS.train_test_ratio == 0:
            test_list.append(stem)
          else:
            train_list.append(stem)
          count += 1
    pickle.dump({'train': train_list, 'test': test_list}, open(FLAGS.list_path, 'wb'))
    glog.info('Create list: train:test=%d:%d.' % (len(train_list), len(test_list)))

  train_numbel = _create_mmc_dataset(vocab, FLAGS.input_dir, train_list, FLAGS.train_output_dir)
  test_numbel = _create_mmc_dataset(vocab, FLAGS.input_dir, test_list, FLAGS.test_output_dir)
  glog.info('Max entity numbel: %d.' % max(train_numbel, test_numbel))


if __name__ == '__main__':
  tf.app.run()

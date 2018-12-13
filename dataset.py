import os
import multiprocessing
import random
import pickle
import time
import brat


class Dataset(multiprocessing.Process):
  def __init__(self, root, max_distance=320, sequence_length=512, buffsize_range=(500, 1000)):
    '''
    Read data to train.

    :param root: Root of dataset, generated from create_datasets.py.
    :param max_distance: Ignore the relation between two entity,
           if the distance between them is greater than max_distance.
    :param sequence_length: Length of clipping the sequence.
    :param buffsize:
    '''

    super(Dataset, self).__init__()
    self._filepaths = []
    self._count = 0
    self._max_distance = max_distance
    self._sequence_length = sequence_length
    for filename in os.listdir(root):
      self._filepaths.append(root + '/' + filename)
    random.shuffle(self._filepaths)
    self._x_positive_buffer = multiprocessing.Manager().list()
    self._y_positive_buffer = multiprocessing.Manager().list()
    self._x_negative_buffer = multiprocessing.Manager().list()
    self._y_negative_buffer = multiprocessing.Manager().list()
    self._buffsize_range = buffsize_range
    self._buffer_lock = multiprocessing.Lock()
    self._requset_stop = multiprocessing.Value('b', False)

  def Start(self):
    self._requset_stop.value = False
    multiprocessing.Process.start(self)

  def Stop(self):
    self._requset_stop.value = True

  def run(self):
    type = 'UP'
    while True:
      if self._requset_stop.value:
        break

      if len(self._y_positive_buffer) > self._buffsize_range[1] and \
          len(self._y_negative_buffer) > self._buffsize_range[1]:
        type = 'Down'
      elif len(self._y_positive_buffer) < self._buffsize_range[0] or \
          len(self._y_negative_buffer) < self._buffsize_range[0]:
        type = 'UP'

      if type == 'UP':
        if self._count >= len(self._filepaths):
          self._count = 0
        if self._count == 0:
          random.shuffle(self._filepaths)
        buff = pickle.load(open(self._filepaths[self._count], 'rb'))
        samples, _ = brat.create_samples(
          buff['words'],
          buff['positions'],
          buff['entities'],
          buff['boxes'],
          max_distance=self._max_distance,
          sequence_length=self._sequence_length)

        self._buffer_lock.acquire(True)
        for sample in samples:
          if sample['y'] == 0:
            self._x_negative_buffer.append(sample['x'])
            self._y_negative_buffer.append(sample['y'])
          else:
            self._x_positive_buffer.append(sample['x'])
            self._y_positive_buffer.append(sample['y'])

        negatives = list(zip(self._x_negative_buffer, self._y_negative_buffer))
        positives = list(zip(self._x_positive_buffer, self._y_positive_buffer))

        random.shuffle(negatives)
        random.shuffle(positives)
        self._x_negative_buffer[:], self._y_negative_buffer[:] = zip(*negatives)
        self._x_positive_buffer[:], self._y_positive_buffer[:] = zip(*positives)
        self._buffer_lock.release()

        self._count += 1
      else:
        time.sleep(1)

  def Get(self, numbel):
    if numbel > self._buffsize_range[0]:
      raise ValueError("numbel must be less than min buffer_size.")
    count = 0
    while True:
      if len(self._y_positive_buffer) > numbel and len(self._y_negative_buffer) > numbel:
        break
      else:
        count += 1
        if count % 10 == 0:
          count = 0
        time.sleep(1)

    pos_totals = len(self._x_positive_buffer)
    neg_totals = len(self._y_negative_buffer)
    num_pos = int(numbel * pos_totals / (pos_totals + neg_totals))
    num_neg = numbel - num_pos
    pos_xs = self._x_positive_buffer[num_pos:]
    pos_ys = self._y_positive_buffer[num_pos:]
    neg_xs = self._x_negative_buffer[num_neg:]
    neg_ys = self._y_negative_buffer[num_neg:]
    return zip(pos_xs, pos_ys), zip(neg_xs, neg_ys)

#  def Evalute(self, s):
#    count = 0
#    size = 0
#
#    # wait len(buffer) > num_select
#    while True:
#      if len(self._x_positive_buffer) > 0:
#        size = int(len(self._y_negative_buffer) * num_pos / len(self._y_positive_buffer))
#      if len(self._y_positive_buffer) > num_pos and size > num_neg:
#        break
#      else:
#        count += 1
#        if count % 10 == 0:
#          count = 0
#        time.sleep(1)
#
#    # sorted negatives
#    pack = list(zip(self._x_negative_buffer[0:size],
#                    self._y_negative_buffer[0:size],
#                    scores_func(self._x_negative_buffer[0:size])))
#    select_size = min(len(pack), self._dilute * num_neg)
#    sorted_pack = list(sorted(pack, key=(lambda i: i[2])))
#    select_pack = sorted_pack[0:select_size]
#    random.shuffle(select_pack)
#    neg_xs, neg_ys, _ = zip(*select_pack)
#    neg_xs, neg_ys = neg_xs[0:num_neg], neg_ys[0:num_neg]
#    pos_xs, pos_ys = self._x_positive_buffer[:num_pos], self._y_positive_buffer[:num_pos]
#
#    self._buffer_lock.acquire(True)
#    if len(self._x_negative_buffer) <= size:
#      self._x_negative_buffer[:] = []
#      self._y_negative_buffer[:] = []
#    else:
#      self._x_negative_buffer[:] = self._x_negative_buffer[size:]
#      self._y_negative_buffer[:] = self._y_negative_buffer[size:]
#    self._x_positive_buffer[:] = self._x_positive_buffer[num_pos:]
#    self._y_positive_buffer[:] = self._y_positive_buffer[num_pos:]
#    self._buffer_lock.release()
#    xs = np.stack(list(pos_xs) + list(neg_xs), axis=0)
#    ys = list(pos_ys) + list(neg_ys)
#    pack = list(zip(xs, ys))
#    random.shuffle(pack)
#

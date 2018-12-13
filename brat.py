import os
import glog
import numpy as np


def Read(filepath):
  stem, ext = os.path.splitext(filepath)

  def parse_ann(lines):
    datas = {'T': {}, 'R': {}}
    for line in lines:
      if len(line) > 0:
        tables = line.strip().split('\t')
        head = tables[0]
        ann_type = head[0]
        ann_numbel = int(head[1:])
        ann = tables[1]
        if ann_type == 'T':
          buff = ann.replace(';', ' ').split(' ')
          label = buff[0]
          position = np.reshape(list(map(int, buff[1:])), [-1, 2]).tolist()
          data = (label, position)
        elif ann_type == 'R':
          buff = ann.split(' ')
          label = buff[0]
          data = (label, [int(buff[1].replace('Arg1:T', '')), int(buff[2].replace('Arg2:T', ''))])
        else:
          glog.error('Unkown line type %s in %s' % (line, filepath))
          continue
        datas[ann_type][ann_numbel] = data
    outputs = {'T': {}, 'R': {}}
    for id in datas['T']:
      poses = datas['T'][id][1]
      for pos in poses:
        length = pos[1] - pos[0]
        if length == 0:
          glog.error('Error length at T%d in %s.' % (id, filepath))
        elif pos[0] < 0 or pos[1] < 0:
          glog.error('Location is out of range at T%d in %s.' % (id, filepath))
        else:
          if length < 0:
            glog.warning('Error location at T%d in %s.' % (id, filepath))
            pos[1], pos[0] = pos[0], pos[1]
          else:
            outputs['T'][id] = datas['T'][id]

    for id in datas['R']:
      arg_id1, arg_id2 = datas['R'][id][1][0], datas['R'][id][1][1]
      if arg_id1 in datas['T'] and arg_id2 in datas['T']:
        outputs['R'][id] = datas['R'][id]
      else:
        glog.warning("Skip R%d, some arg was not found at %s." % (id, filepath))
    return outputs

  def parse_txt(lines):
    result = []
    for line in lines:
      for word in line:
        result.append(word)
    return result

  parser = {'.ann': parse_ann, '.txt': parse_txt}

  if ext in parser:
    lines = open(filepath, 'r', encoding='utf-8').readlines()
    return parser[ext](lines)
  else:
    glog.error('Unkown file type: %s' % filepath)


def split(inputs, max_length=168, boundary=20):
  inputs = boundary * [0] + inputs + boundary * [0]
  size = len(inputs)
  start = 0
  datas = []
  while True:
    if start >= size:
      break
    input = [0] * max_length
    start = start - 2 * boundary
    if start < 0:
      start = 0
    end = size - start
    for i in range(min(max_length, end)):
      input[i] = inputs[start + i]
    datas.append(tuple(input))
    start = start + max_length
  return datas


def splits(words, entities, max_length=168, boundary=20):
  size = min(len(words), len(entities))
  words = split(words[0:size], max_length, boundary)
  entities = split(entities[0:size], max_length, boundary)
  datas = []
  for i in range(min(len(words), len(entities))):
    datas.append((words[i], entities[i]))
  return datas


def merge(inputs, boundary=20):
  overlap = 2 * boundary
  outputs = []
  for input in inputs:
    if len(outputs) < overlap:
      outputs += input[boundary:]
    else:
      if boundary > 0:
        outputs[- boundary:] = input[boundary: overlap]
      outputs += input[overlap:]
  return outputs


def merges(words, entities, boundary=20):
  words = merge(words, boundary)
  entities = merge(entities, boundary)
  return words, entities


def Encode(words, anns, vocab):
  size = len(words)
  enocde_words = np.zeros(size, dtype=np.uint16)
  for i, word in enumerate(words):
    if word in vocab['Word']:
      enocde_words[i] = vocab['Word'].index(word)
    else:
      enocde_words[i] = 0
  if anns is None:
    return enocde_words

  encode_entities = np.zeros(size, dtype=np.uint8)
  for ann in anns['T'].values():
    for p in ann[1]:
      encode_entities[p[0]: p[1]] = vocab['Entity'].index(ann[0])

  enocde_position = np.zeros(size, dtype=np.uint16)
  sorted_entity = sorted(anns['T'].items(), key=lambda item: item[1][1][0][0])
  for i, item, in enumerate(sorted_entity):
    enocde_position[item[1][1][0][0]:item[1][1][-1][-1]] = i + 1

  encode_relations = {}
  for key in anns['R']:
    relation = anns['R'][key]
    label = vocab['Relation'].index(relation[0])
    arg1, arg2 = relation[1]
    pos1 = anns['T'][arg1]
    pos2 = anns['T'][arg2]
    encode_relations[key] = {'minx': pos2[1][0][0],
                             'miny': pos1[1][0][0],
                             'maxx': pos2[1][-1][-1],
                             'maxy': pos1[1][-1][-1],
                             'label': label,
                             'key1': arg1,
                             'key2': arg2}

  return enocde_words, enocde_position, encode_entities, encode_relations


def Decode(words, entities, relations, vocab):
  result = {}
  size = len(words)

  if isinstance(words[int(size / 2)], int):
    encode_words = words
    words = []
    for word in encode_words:
      words.append(vocab['Word'][word])

  size = min(size, len(entities))

  def decode_sequence(sequence):
    lines = []
    index = 0
    indexes = np.zeros(size, np.uint16)
    while True:
      if index >= size:
        break
      start_label = sequence[index]
      if start_label != 0 and words[index] != '\n':
        line = [vocab['Entity'][start_label], []]
        start = index
        while True:
          index += 1
          if index >= size:
            end = index
            line[1].append((start, end))
            break
          end_label = sequence[index]
          if end_label != start_label or words[index] == '\n':
            end = index
            line[1].append((start, end))

            if words[index] != '\n' and end_label != start_label:
              break
            while True:
              if index >= size:
                break
              if words[index] != '\n':
                start = index
                break
              index += 1
            if sequence[start] != start_label:
              break

        lines.append(line)
      else:
        index += 1

    for i, line in enumerate(lines):
      for p in line[1]:
        indexes[p[0]:p[1]] = i + 1
    return lines, indexes

  decode_labels, indexes = decode_sequence(entities)
  decode_relations = {}

  for rect in relations:
    def get_relation_index(min_ids, max_ids):
      tmp = {}
      for i in range(min_ids, max_ids):
        index = indexes[i]
        if index == 0:
          continue
        if index in tmp:
          tmp[index] += 1
        else:
          tmp[index] = 1
      max_score, max_index = 0, 0
      for key in tmp:
        if tmp[key] >= max_score:
          max_index = key
          max_score = tmp[key]
      return max_index

    arg2 = get_relation_index(rect['minx'], rect['maxx'])
    arg1 = get_relation_index(rect['miny'], rect['maxy'])
    if arg2 != 0 and arg1 != 0:
      decode_relations[(arg1, arg2)] = vocab['Relation'][rect['label']]

  result['T'] = decode_labels
  result['R'] = []
  for key in decode_relations:
    result['R'].append([decode_relations[key], [key[0], key[1]]])
  return words, result


def pack_boxes(encode_words, encode_positions, encode_entities, boxes, sequence_length=128):
  def clip(start, end):
    i_start = int((start + end - sequence_length + 1) / 2)

    o_start = start - i_start
    o_end = o_start + (end - start)

    inputs = np.zeros([sequence_length], dtype=np.uint16)
    positions = np.zeros([sequence_length], dtype=np.uint16)
    type_ids = np.zeros([sequence_length], dtype=np.uint16)
    masks = np.zeros([sequence_length], dtype=np.uint16)

    for i in range(0, sequence_length):
      id = i_start + i
      if id in range(0, len(encode_words)):
        inputs[i] = encode_words[id]
        positions[i] = encode_positions[id]
        type_ids[i] = encode_entities[id]

    for i in range(o_start, o_end):
      masks[i] = 1
    return np.stack([inputs, positions, type_ids, masks], axis=0)

  outputs = []
  for box in boxes:
    data1 = clip(box['miny'], box['maxy'])
    data2 = clip(box['minx'], box['maxx'])
    outputs.append({'x': np.stack((data1, data2), axis=0), 'y': box['label']})
  return outputs


def create_samples(encode_words, encode_positions, encode_entities, boxes, max_distance=356, sequence_length=512):
  def clip(start1, end1, start2, end2):
    if start1 == start2 and end1 == end2:
      return None

    min_value = min(start1, start2)
    max_value = max((end1, end2))
    if max_value - min_value > max_distance:
      return None

    position1 = np.mean(encode_positions[start1:end1])
    position2 = np.mean(encode_positions[start2:end2])

    i_start = int((min_value + max_value - sequence_length + 1) / 2)
    o_start1 = start1 - i_start
    o_start2 = start2 - i_start
    o_end1 = o_start1 + end1 - start1
    o_end2 = o_start2 + end2 - start2

    inputs = np.zeros([sequence_length], dtype=np.uint16)
    positions = np.zeros([sequence_length], dtype=np.uint16)
    type_ids = np.zeros([sequence_length], dtype=np.uint16)
    masks = np.zeros([sequence_length], dtype=np.uint16)

    positions[o_start1:o_end1] = position1
    positions[o_start2:o_end2] = position2

    for i in range(0, sequence_length):
      id = i_start + i
      if id in range(0, len(encode_words)):
        inputs[i] = encode_words[id]

        if encode_entities[id] > 0:
          type_ids[i] = encode_entities[id]

        if position1 < position2:
          position = encode_positions[id] - position1 + 1
          if position > 0:
            positions[i] = position
        else:
          if encode_positions[id] != 0:
            position = position1 - encode_positions[id] + 1
            if position > 0:
              positions[i] = position

    for i in range(o_start1, o_end1):
      masks[i] = 1
    for i in range(o_start2, o_end2):
      masks[i] = 1
    return np.stack([inputs, positions, type_ids, masks], axis=0)

  output_samples = []
  output_boxes = []
  for box in boxes:
    data = clip(box['miny'], box['maxy'], box['minx'], box['maxx'])
    if data is not None:
      if box['miny'] <= box['minx'] and box['maxy'] <= box['maxx']:
        direction = 0
      elif box['miny'] >= box['minx'] and box['maxy'] >= box['maxx']:
        direction = 1
      else:
        glog.fatal('Unkown error of box(%d, %d, %d, %d).' % (box['minx'], box['miny'], box['maxx'], box['maxy']))
        continue
      if 'label' in box:
        output_samples.append({'x': data, 'd': direction, 'y': box['label']})
      else:
        output_samples.append({'x': data, 'd': direction})
      output_boxes.append(box)
  return output_samples, output_boxes


#  def splite_boxes(encode_words, encode_positions, encode_entities, boxes, overlap=128, sequence_length=512):
#    outputs = []
#
#    start = 0
#    while True:
#      splite_word = np.zeros()
#      end = min(start + sequence_length, len(encode_words))
#      relation.ioas(boxes, {'minx': start, 'miny': end, 'maxx': start, 'maxy': end})
#
#      start = end - overlap
#
#    def clip(start1, end1, start2, end2):
#      if start1 == start2 and end1 == end2:
#        return None
#
#      min_value = min(start1, start2)
#      max_value = max((end1, end2))
#      if max_value - min_value > max_length:
#        return None
#
#      i_start = int((min_value + max_value - sequence_length + 1) / 2)
#      o_start1 = start1 - i_start
#      o_start2 = start2 - i_start
#      o_end1 = o_start1 + end1 - start1
#      o_end2 = o_start2 + end2 - start2
#
#      inputs = np.zeros([sequence_length], dtype=np.uint16)
#      positions = np.zeros([sequence_length], dtype=np.uint16)
#      type_ids = np.zeros([sequence_length], dtype=np.uint16)
#      masks = np.zeros([sequence_length], dtype=np.uint16)
#
#      start = 1
#      for i in range(0, sequence_length):
#        id = i_start + i
#        if id in range(0, len(encode_words)):
#          inputs[i] = encode_words[id]
#          type_ids[i] = encode_entities[id]
#          if encode_positions[id] > 0:
#            start = min(encode_positions[id], start)
#
#      for i in range(0, sequence_length):
#        id = i_start + i
#        if id in range(0, len(encode_words)):
#          if encode_positions[id] > 0:
#            positions[i] = encode_positions[id] - start + 1
#
#      for i in range(o_start1, o_end1):
#        masks[i] = 1
#      for i in range(o_start2, o_end2):
#        masks[i] = 1
#      return np.stack([inputs, positions, type_ids, masks], axis=0)
#
#  outputs = []
#  for box in boxes:
#    data = clip(box['miny'], box['maxy'], box['minx'], box['maxx'])
#    if data is not None:
#      if box['miny'] <= box['minx'] and box['maxy'] <= box['maxx']:
#        direction = 0
#      elif box['miny'] > box['minx'] and box['maxy'] > box['maxx']:
#        direction = 1
#      else:
#        glog.FATAL('Unkown error.')
#      outputs.append({'x': data, 'd': direction, 'y': box['label']})
#  return outputs


def Write(filepath, words, anns):
  file = open(filepath, 'w', encoding='utf-8')
  for id in anns['T']:
    line = anns['T'][id]
    txt = 'T%d\t%s ' % (id, line[0])
    for j, p in enumerate(line[1]):
      if j != 0:
        txt += ';'
      txt += '%d %d' % (p[0], p[1])
    txt += '\t'
    for j, p in enumerate(line[1]):
      if j != 0:
        txt += ' '
      txt += "".join(words[p[0]: p[1]])
    txt += '\n'
    file.writelines(txt)

  for id in anns['R']:
    line = anns['R'][id]
    txt = 'R%d\t%s Arg1:T%d Arg2:T%d\n' % (id, line[0], line[1][0], line[1][1])
    file.writelines(txt)

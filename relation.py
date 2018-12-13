import glog
import numpy as np

relation_dict = {'Test': 'Disease',
                 'Treatment': 'Disease',
                 'Symptom': 'Disease',
                 'Drug': 'Disease',
                 'Anatomy': 'Disease',
                 'Frequency': 'Drug',
                 'Duration': 'Drug',
                 'Amount': 'Drug',
                 'Method': 'Drug',
                 'SideEff': 'Drug'}


def area(box):
  width = box['maxx'] - box['minx']
  height = box['maxy'] - box['miny']
  area = width * height
  if width < 0 or height < 0:
    return 0 - abs(area)
  return area


def create(entities, filter=True):
  boxes = []
  for key1 in entities:
    entity1 = entities[key1]
    for key2 in entities:
      entity2 = entities[key2]
      if filter:
        if entity1[0] not in relation_dict or entity2[0] != relation_dict[entity1[0]]:
          continue

      box = {'minx': entity2[1][0][0],
             'miny': entity1[1][0][0],
             'maxx': entity2[1][-1][-1],
             'maxy': entity1[1][-1][-1],
             'key1': key1,
             'key2': key2}
      if area(box) <= 0:
        glog.error('Unkwon error.')
        continue
      boxes.append(box)
  return boxes


def _area(boxes):
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _encode_boxes(boxes):
  size = len(boxes)
  outputs = np.zeros([size, 4], dtype=np.float32)
  for i, boxes in enumerate(boxes):
    outputs[i] = [boxes['miny'], boxes['minx'], boxes['maxy'], boxes['maxx']]
  return outputs


def intersection(boxes1, boxes2):
  [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
  [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

  all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
  all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
  intersect_heights = np.maximum(
    np.zeros(all_pairs_max_ymin.shape, np.float32),
    all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
  all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
  intersect_widths = np.maximum(
    np.zeros(all_pairs_max_xmin.shape, np.float32),
    all_pairs_min_xmax - all_pairs_max_xmin)
  return intersect_heights * intersect_widths


def ious(boxes1, boxes2):
  boxes1 = _encode_boxes(boxes1)
  boxes2 = _encode_boxes(boxes2)
  intersect = intersection(boxes1, boxes2)
  area1 = _area(boxes1)
  area2 = _area(boxes2)
  union = np.expand_dims(area1, axis=1) + np.expand_dims(
    area2, axis=0) - intersect
  return intersect / union


def ioas(boxes1, boxes2):
  boxes1 = _encode_boxes(boxes1)
  boxes2 = _encode_boxes(boxes2)
  intersect = intersection(boxes1, boxes2)
  areas = np.expand_dims(_area(boxes2), axis=0)
  return intersect / areas

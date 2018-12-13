import six
import json
import copy
import tensorflow as tf


class Base:
  def __init__(self):
    pass

  def __str__(self):
    buff = ''
    for key, var in vars(self).items():
      buff += "%s: %s.\n" % (key, var)
    return buff

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = cls()
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

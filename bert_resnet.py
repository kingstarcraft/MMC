'''
Paper: Deep Residual Learning for Weakly-Supervised Relation Extraction
'''

import models.bert as bert
import tensorflow as tf
import models.resnet as resnet


def bulid_bert_resnet_model(vocab, inputs, is_training, bert_config=None, resnet_config=None):
  if bert_config is None:
    bert_config = bert.BertConfig(len(vocab['Word']),
                                  max_position_embeddings=512,
                                  type_vocab_size=len(vocab['Entity']))
  if resnet_config is None:
    resnet_config = resnet.ResnetConfig()

  def splits(datas, num, name='split_squeeze'):
    with tf.variable_scope(name):
      datas = tf.split(datas, num, axis=1)
      outputs = []
      for data in datas:
        outputs.append(tf.squeeze(data, axis=1))
      return outputs

  input_ids, token_position_ids, token_type_ids, input_mask = splits(inputs, 4)
  model = bert.BertModel(bert_config, is_training, input_ids,
                         input_mask=input_mask,
                         token_type_ids=token_type_ids,
                         token_position_ids=token_position_ids)
  bert_features = model.get_sequence_output()
  resnet_features = resnet.resnet(bert_features, resnet_config)
  features = tf.reduce_max(resnet_features, axis=1)  # max_pool
  logits = tf.layers.dense(features, len(vocab['Relation']), name='logits')
  return logits

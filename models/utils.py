import six
import tensorflow as tf


def gelu(input_tensor):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)

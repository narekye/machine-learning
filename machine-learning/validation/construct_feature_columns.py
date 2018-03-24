import tensorflow as tf

def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
    return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
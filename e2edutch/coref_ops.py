from tensorflow.python import pywrap_tensorflow


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

coref_op_library = tf.load_op_library("./lib/coref_kernels.so")

extract_spans = coref_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")

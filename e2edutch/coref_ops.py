from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python import pywrap_tensorflow

coref_op_library = tf.load_op_library("./lib/coref_kernels.so")

extract_spans = coref_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")

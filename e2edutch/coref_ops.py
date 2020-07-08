from tensorflow.python import pywrap_tensorflow
import pkg_resources

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

coref_op_library = tf.load_op_library(
                    pkg_resources.resource_filename("e2edutch",
                                                    "lib/coref_kernels.so"))

extract_spans = coref_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")

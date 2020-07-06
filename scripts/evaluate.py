#!/usr/bin/env python
from __future__ import absolute_import
from e2edutch import util
from e2edutch import coref_model as cm
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

if __name__ == "__main__":
    config = util.initialize_from_env(sys.argv[1])
    model = cm.CorefModel(config)
    include_singletons = config['include_singletons']
    with tf.Session() as session:
        model.restore(session)
        model.evaluate(session, official_stdout=True,
                       include_singletons=include_singletons)

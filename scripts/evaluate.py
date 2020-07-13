#!/usr/bin/env python
from e2edutch import util
from e2edutch import coref_model as cm

import os
import sys

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

if __name__ == "__main__":
    config = util.initialize_from_env(sys.argv[1])
    model = cm.CorefModel(config)
    with tf.Session() as session:
        model.restore(session)
        model.evaluate(session, official_stdout=True)

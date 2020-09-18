#!/usr/bin/env python
from e2edutch import util
from e2edutch import coref_model as cm

import os
import sys
import argparse

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--cfg_file',
        type=str,
        default=None,
        help="config file")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    config = util.initialize_from_env(args.config, args.cfg_file)
    model = cm.CorefModel(config)
    with tf.Session() as session:
        model.restore(session)
        model.evaluate(session, official_stdout=True)

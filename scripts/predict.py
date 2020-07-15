from e2edutch import conll
from e2edutch import minimize
from e2edutch import util
from e2edutch import coref_model as cm

import sys
import json
import os
import collections
import argparse
import logging

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('input_filename')
    parser.add_argument('-o', '--output_file',
                        type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('-f', '--format_out', default='conll',
                        choices=['conll', 'jsonlines'])
    parser.add_argument('-c', '--word_col', type=int, default=2)
    return parser


def read_jsonlines(input_filename):
    for line in open(input_filename).readlines():
        example = json.loads(line)
        yield example


def main(args=None):
    parser = get_parser()
    args = parser.parse_args()
    config = util.initialize_from_env(args.config)

    # Input file in .jsonlines format or .conll.
    input_filename = args.input_filename

    ext_input = os.path.splitext(input_filename)[-1]
    if ext_input not in ['.conll', '.jsonlines', '.txt']:
        raise Exception(
            'Input file should be .conll, .txt or .jsonlines, but is {}.'.format(ext_input))

    if ext_input == '.conll':
        labels = collections.defaultdict(set)
        stats = collections.defaultdict(int)
        docs = minimize.minimize_partition(
            input_filename, labels, stats, args.word_col)
    elif ext_input == '.jsonlines':
        docs = read_jsonlines(input_filename)
    else:
        text = open(input_filename).read()
        docs = [util.create_example(text)]

    output_file = args.output_file
    model = cm.CorefModel(config)
    sentences = {}
    predictions = {}
    with tf.Session() as session:
        model.restore(session)
        for example_num, example in enumerate(docs):
            # logging.info(example['doc_key'])
            tensorized_example = model.tensorize_example(
                example, is_training=False)
            feed_dict = {i: t for i, t in zip(
                model.input_tensors, tensorized_example)}
            _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
                model.predictions, feed_dict=feed_dict)
            predicted_antecedents = model.get_predicted_antecedents(
                top_antecedents, top_antecedent_scores)
            example["predicted_clusters"], _ = model.get_predicted_clusters(
                top_span_starts, top_span_ends, predicted_antecedents)
            if args.format_out == 'jsonlines':
                output_file.write(json.dumps(example))
                output_file.write("\n")
            else:
                predictions[example['doc_key']] = example["predicted_clusters"]
                sentences[example['doc_key']] = example["sentences"]
            if example_num % 100 == 0:
                logging.info("Decoded {} examples.".format(example_num + 1))
        if args.format_out == 'conll':
            conll.output_conll(output_file, sentences, predictions)


if __name__ == "__main__":
    main()

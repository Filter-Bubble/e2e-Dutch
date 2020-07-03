from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os
import collections

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from e2edutch import coref_model as cm
from e2edutch import util
from e2edutch import minimize

def read_jsonlines(input_filename):
    for line in open(input_filename).readlines():
        example = json.loads(line)
        yield example


if __name__ == "__main__":
    config = util.initialize_from_env(sys.argv[1])

    # Input file in .jsonlines format.
    input_filename = sys.argv[2]

    ext_input = os.path.splitext(input_filename)[-1]
    if ext_input not in ['.conll', '.jsonlines']:
        raise Exception('Input file should be either .conll or .jsonlines, but is {}.'.format(ext_input))

    if ext_input == '.conll':
        labels = collections.defaultdict(set)
        stats = collections.defaultdict(int)
        docs = minimize.minimize_partition(input_filename, labels, stats)
    else:
        docs = read_jsonlines(input_filename)

    # Predictions will be written to this file in .jsonlines format.
    # To do: add possibility of outputting conll
    output_filename = sys.argv[3]
    ext_output = os.path.splitext(output_filename)[-1]
    if ext_output not in ['.conll', '.jsonlines']:
        raise Exception('Output file should be either .conll or .jsonlines, but is {}.'.format(ext_output))

    model = cm.CorefModel(config)
    include_singletons = config['include_singletons']
    with tf.Session() as session:
        model.restore(session)

        with open(output_filename, "w") as output_file:
            for example_num, example in enumerate(docs):
                print(example['doc_key'])
                tensorized_example = model.tensorize_example(
                    example, is_training=False)
                feed_dict = {i: t for i, t in zip(
                    model.input_tensors, tensorized_example)}
                _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
                    model.predictions, feed_dict=feed_dict)
                predicted_antecedents = model.get_predicted_antecedents(
                    top_antecedents, top_antecedent_scores)
                example["predicted_clusters"], _ = model.get_predicted_clusters(
                    top_span_starts, top_span_ends, predicted_antecedents,
                    include_singletons=include_singletons)

                output_file.write(json.dumps(example))
                output_file.write("\n")
                if example_num % 100 == 0:
                    print("Decoded {} examples.".format(example_num + 1))

import sys
import json
import os
import io
import collections
import argparse
import logging

from e2edutch import conll
from e2edutch import minimize
from e2edutch import util
from e2edutch import coref_model as cm
from e2edutch import naf

import tensorflow.compat.v1 as tf

logger = logging.getLogger('e2edutch')


class Predictor(object):
    """
    A predictor object loads a pretrained e2e model to predict coreferences.
    It can be used to predict coreferences on tokenized text.
    """

    def __init__(self, model_name='final', config=None, verbose=False):
        if verbose:
            logger.setLevel(logging.INFO)

        if config:
            self.config = config
        else:
            # if no configuration is provided, try to get a default config.
            self.config = util.initialize_from_env(model_name=model_name)

        # Clear tensorflow context:
        tf.reset_default_graph()
        self.session = tf.compat.v1.Session()

        try:
            self.model = cm.CorefModel(self.config)
            self.model.restore(self.session)
        except ValueError:
            raise Exception("Trying to reload the model while the previous " +
                            "session hasn't been ended. Close the existing " +
                            "session with predictor.end_session()")

    def predict(self, example):
        """
        Predict coreference spans for a tokenized text.


        Args:
            example (dict): dict with the following fields:
                              sentences ([[str]])
                              doc_id (str)
                              clusters ([[(int, int)]]) (optional)

        Returns:
            [[(int, int)]]: a list of clusters. The items of the cluster are
                            spans, denoted by their start end end token index

        """
        tensorized_example = self.model.tensorize_example(
            example, is_training=False)
        feed_dict = {i: t for i, t in zip(
            self.model.input_tensors, tensorized_example)}
        _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = self.session.run(
            self.model.predictions, feed_dict=feed_dict)
        predicted_antecedents = self.model.get_predicted_antecedents(
            top_antecedents, top_antecedent_scores)
        predicted_clusters, _ = self.model.get_predicted_clusters(
            top_span_starts, top_span_ends, predicted_antecedents)

        return predicted_clusters

    def end_session(self):
        """
        Close the session, clearing the tensorflow model context.
        """
        self.session.close()
        tf.reset_default_graph()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename')
    parser.add_argument('-o', '--output_file',
                        type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('-f', '--format_out', default='conll',
                        choices=['conll', 'jsonlines', 'naf'])
    parser.add_argument('-m', '--model',
                        type=str,
                        default='final',
                        help="model name")
    parser.add_argument('-c', '--word_col', type=int, default=2)
    parser.add_argument('--cfg_file',
                        type=str,
                        default=None,
                        help="config file")
    parser.add_argument('--model_cfg_file',
                        type=str,
                        default=None,
                        help="model config file")
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def read_jsonlines(input_filename):
    for line in open(input_filename).readlines():
        example = json.loads(line)
        yield example


def main(args=None):
    parser = get_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.INFO)

    # Input file in .jsonlines format or .conll.
    input_filename = args.input_filename

    ext_input = os.path.splitext(input_filename)[-1]
    if ext_input not in ['.conll', '.jsonlines', '.txt', '.naf']:
        raise Exception(
            'Input file should be .naf, .conll, .txt or .jsonlines, but is {}.'
            .format(ext_input))

    if ext_input == '.conll':
        labels = collections.defaultdict(set)
        stats = collections.defaultdict(int)
        docs = minimize.minimize_partition(
            input_filename, labels, stats, args.word_col)
    elif ext_input == '.jsonlines':
        docs = read_jsonlines(input_filename)
    elif ext_input == '.naf':
        naf_obj = naf.get_naf(input_filename)
        jsonlines_obj, term_ids, tok_ids = naf.get_jsonlines(naf_obj)
        docs = [jsonlines_obj]
    else:
        text = open(input_filename).read()
        docs = [util.create_example(text)]

    output_file = args.output_file

    config = util.initialize_from_env(model_name=args.model,
                                      cfg_file=args.cfg_file,
                                      model_cfg_file=args.model_cfg_file)
    predictor = Predictor(config=config)

    sentences = {}
    predictions = {}
    for example_num, example in enumerate(docs):
        example["predicted_clusters"] = predictor.predict(example)
        if args.format_out == 'jsonlines':
            output_file.write(json.dumps(example))
            output_file.write("\n")
        else:
            predictions[example['doc_key']] = example["predicted_clusters"]
            sentences[example['doc_key']] = example["sentences"]
        if example_num % 100 == 0:
            logger.info("Decoded {} examples.".format(example_num + 1))
    if args.format_out == 'conll':
        conll.output_conll(output_file, sentences, predictions)
    elif args.format_out == 'naf':
        # Check number of docs - what to do if multiple?
        # Create naf obj if input format was not naf
        if ext_input != '.naf':
            # To do: add linguistic processing layers for terms and tokens
            logger.warn(
                'Outputting NAF when input was not naf,'
                + 'no dependency information available')
            for doc_key in sentences:
                naf_obj, term_ids = naf.get_naf_from_sentences(
                    sentences[doc_key])
                naf_obj = naf.create_coref_layer(
                    naf_obj, predictions[doc_key], term_ids)
                naf_obj = naf.add_linguistic_processors(naf_obj)
                buffer = io.BytesIO()
                naf_obj.dump(buffer)
                output_file.write(buffer.getvalue().decode('utf-8'))
                # To do, make sepearate outputs?
                # TO do, use dependency information from conll?
        else:
            # We only have one input doc
            naf_obj = naf.create_coref_layer(
                naf_obj, example["predicted_clusters"], term_ids)
            naf_obj = naf.add_linguistic_processors(naf_obj)
            buffer = io.BytesIO()
            naf_obj.dump(buffer)
            output_file.write(buffer.getvalue().decode('utf-8'))


if __name__ == "__main__":
    main()

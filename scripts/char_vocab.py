from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import argparse
import logging


def get_char_vocab(input_filenames, output_file):
    vocab = set()
    for filename in input_filenames:
        with open(filename) as f:
            for line in f.readlines():
                for sentence in json.loads(line)["sentences"]:
                    for word in sentence:
                        vocab.update(word)
    vocab = sorted(list(vocab))
    for char in vocab:
        output_file.write(char)
        output_file.write(u"\n")
    logging.info("Wrote {} characters".format(len(vocab)))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filenames', nargs='+')
    parser.add_argument('output_file',
                        type=argparse.FileType('w'))
    return parser


if __name__=="__main__":
    args = get_parser().parse_args()
    get_char_vocab(args.input_filenames, args.output_file)

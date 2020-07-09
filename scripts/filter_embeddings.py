from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import argparse
import logging

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('embedding_file',
                        type=argparse.FileType('r'))
    parser.add_argument('-c', '--cased', default=False)
    parser.add_argument('doc_files', nargs='+')
    parser.add_argument('out_file',
                        type=argparse.FileType('w'))
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    words_to_keep = set()
    for json_filename in args.doc_files:
        with open(json_filename) as json_file:
            for line in json_file.readlines():
                for sentence in json.loads(line)["sentences"]:
                    if args.cased:
                        words_to_keep.update(sentence)
                    else:
                        words_to_keep.update([w.lower() for w in sentence])

    logging.info("Found {} words in {} dataset(s).".format(
        len(words_to_keep), len(sys.argv) - 3))

    total_lines = 0
    kept_lines = 0
    for line in args.embedding_file.readlines():
            total_lines += 1
            word = line.split()[0]
            if word in words_to_keep:
                kept_lines += 1
                args.out_file.write(line)

    logging.info("Kept {} out of {} lines.".format(kept_lines, total_lines))

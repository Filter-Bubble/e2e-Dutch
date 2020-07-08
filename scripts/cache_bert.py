import numpy as np
import h5py
import json
import logging
import argparse

from e2edutch import bert


def cache_dataset(data_path, out_file, tokenizer, model):
    with open(data_path) as in_file:
        for doc_num, line in enumerate(in_file.readlines()):
            example = json.loads(line)
            sentences = example["sentences"]
            bert_final = bert.encode_sentences(sentences, tokenizer, model)
            # shape: (num_sent, max_sent_len, lm_size, 1)
            text_len = np.array([len(s) for s in sentences])
            file_key = example["doc_key"].replace("/", ":")
            if file_key in out_file.keys():
                del out_file[file_key]

            group = out_file.create_group(file_key)
            for i, (e, l) in enumerate(zip(bert_final, text_len)):
                e = np.array(e[:l, :, :])
                group[str(i)] = e
            if doc_num % 10 == 0:
                logging.info("Cached {} documents in {}".format(
                    doc_num + 1, data_path))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', choices=['bertje', 'bert-nl', 'robbert'])
    parser.add_argument('input_files', nargs='+')
    return parser


def main(args=None):
    args = get_parser().parse_args()
    model_name = args.model_name
    tokenizer, model = bert.load_bert(model_name)
    with h5py.File("data/{}_cache.hdf5".format(model_name), "a") as out_file:
        for json_filename in args.input_files:
            cache_dataset(json_filename, out_file, tokenizer, model)


if __name__ == "__main__":
    main()

import logging
import argparse
import e2edutch.util
import e2edutch.coref_model as cm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def print_predictions(example):
    words = e2edutch.util.flatten(example["sentences"])
    for cluster in example["predicted_clusters"]:
        print(u"Predicted cluster: {}".format(
            [" ".join(words[m[0]:m[1] + 1]) for m in cluster]))


def make_predictions(text, model):
    example = e2edutch.util.create_example(text)
    tensorized_example = model.tensorize_example(example, is_training=False)
    feed_dict = {i: t for i, t in zip(model.input_tensors, tensorized_example)}
    _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(
        model.predictions + [model.head_scores], feed_dict=feed_dict)

    predicted_antecedents = model.get_predicted_antecedents(
        antecedents, antecedent_scores)

    example["predicted_clusters"], _ = model.get_predicted_clusters(
        mention_starts, mention_ends, predicted_antecedents)
    example["top_spans"] = zip(
        (int(i) for i in mention_starts), (int(i) for i in mention_ends))
    example["head_scores"] = head_scores.tolist()
    return example


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    # , default=sys.stdin)
    parser.add_argument('input_file', type=argparse.FileType('r'))
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


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    config = e2edutch.util.initialize_from_env(
        args.config, args.cfg_file, args.model_cfg_file)
    model = cm.CorefModel(config)
    with tf.Session() as session:
        model.restore(session)
        text = args.input_file.read()
        if len(text) > 0:
            print_predictions(make_predictions(text, model))

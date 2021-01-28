#!/usr/bin/env python
import os
import time
import argparse
import logging
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from e2edutch import util
from e2edutch import coref_model as cm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--train',
                        type=str,
                        default=None,
                        help="jsonlines file used for training")
    parser.add_argument('--eval',
                        type=str,
                        default=None,
                        help="jsonlines file used for evaluating")
    parser.add_argument('--eval_conll',
                        type=str,
                        default=None,
                        help="conll file used for evaluating")
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


def main(args=None):
    args = get_parser().parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    config = util.initialize_from_env(args.config,
                                      args.cfg_file,
                                      args.model_cfg_file)
    # Overwrite train and eval file if specified
    if args.train is not None:
        config['train_path'] = args.train
    if args.eval is not None:
        config['eval_path'] = args.eval
    if args.eval_conll is not None:
        config['conll_eval_path'] = args.eval_conll

    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]

    model = cm.CorefModel(config)
    saver = tf.train.Saver()

    log_dir = os.path.join(config['log_root'], config['log_dir'])
    writer = tf.summary.FileWriter(log_dir, flush_secs=20)

    max_f1 = 0

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.start_enqueue_thread(session)
        accumulated_loss = 0.0

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt.model_checkpoint_path))
            saver.restore(session, ckpt.model_checkpoint_path)

        initial_time = time.time()
        while True:
            tf_loss, tf_global_step, _ = session.run(
                [model.loss, model.global_step, model.train_op])
            accumulated_loss += tf_loss

            if tf_global_step % report_frequency == 0:
                total_time = time.time() - initial_time
                steps_per_second = tf_global_step / total_time

                average_loss = accumulated_loss / report_frequency
                print("[{}] loss={:.2f}, steps/s={:.2f}"
                      .format(tf_global_step,
                              average_loss,
                              steps_per_second))
                writer.add_summary(util.make_summary(
                    {"loss": average_loss}), tf_global_step)
                accumulated_loss = 0.0

            if tf_global_step % eval_frequency == 0:
                saver.save(session, os.path.join(log_dir, "model"),
                           global_step=tf_global_step)
                eval_summary, eval_f1 = model.evaluate(session)

                if eval_f1 > max_f1:
                    max_f1 = eval_f1
                    util.copy_checkpoint(os.path.join(
                        log_dir, "model-{}".format(tf_global_step)),
                                         os.path.join(log_dir, "model.max.ckpt"))

                writer.add_summary(eval_summary, tf_global_step)
                writer.add_summary(util.make_summary(
                    {"max_eval_f1": max_f1}), tf_global_step)

                print("[{}] evaL_f1={:.2f}, max_f1={:.2f}".format(
                    tf_global_step, eval_f1, max_f1))


if __name__ == "__main__":
    main()

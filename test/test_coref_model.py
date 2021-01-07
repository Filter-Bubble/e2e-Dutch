import tensorflow.compat.v1 as tf
from collections import OrderedDict
import e2edutch.coref_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.disable_v2_behavior()

test_config = {'max_top_antecedents': 30,
               'max_training_sentences': 30,
               'top_span_ratio': 0.4,
               'filter_widths': [3, 4, 5],
               'filter_size': 50,
               'char_embedding_size': 8,
               'char_vocab_path': 'data/char_vocab.dutch.txt',
               'context_embeddings': OrderedDict([('path', 'test/data/test-embeddings.vec'),
                                                  ('size', 10),
                                                  ('cased', True)]),
               'head_embeddings': OrderedDict([('path', 'test/data/test-embeddings.vec'),
                                               ('size', 10),
                                               ('cased', True)]),
               'contextualization_size': 200,
               'contextualization_layers': 3,
               'ffnn_size': 150,
               'ffnn_depth': 2,
               'feature_size': 20,
               'max_span_width': 30,
               'use_metadata': True,
               'use_features': True,
               'model_heads': True,
               'coref_depth': 2,
               'lm_path': None,
               'lm_model_name': None,
               'lm_layers': 1,
               'lm_size': 768,
               'coarse_to_fine': True,
               'max_gradient_norm': 5.0,
               'lstm_dropout_rate': 0.4,
               'lexical_dropout_rate': 0.5,
               'dropout_rate': 0.2,
               'optimizer': 'adam',
               'learning_rate': 0.001,
               'decay_rate': 0.999,
               'decay_frequency': 100,
               'genres': ['all'],
               'eval_frequency': 5000,
               'report_frequency': 100,
               'use_gold': False,
               'log_root': 'logs',
               'log_dir': 'logs/best',
               'datapath': '.'
               }


def test_tensorize_example():
    example = {
        'doc_key': 'test',
        'sentences': [['Een', 'zin', '.']],
        'clusters': []
    }
    with tf.Session() as session:
        model = e2edutch.coref_model.CorefModel(test_config)
        tensors = model.tensorize_example(example, False)
        (tokens, context_word_emb, head_word_emb, lm_emb,
         char_index, text_len, genre, is_training,
         gold_starts, gold_ends, cluster_ids) = tensors
    tf.reset_default_graph()
    assert tokens.shape == (1, 3)
    assert context_word_emb.shape == (1, 3, 10)
    assert head_word_emb.shape == (1, 3, 10)
    assert lm_emb.shape == (0, 0,
                            test_config['lm_size'], test_config['lm_layers'])
    assert len(gold_starts) == 0

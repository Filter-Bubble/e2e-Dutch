import e2edutch.coref_model_new
import tensorflow as tf
import numpy as np
from collections import OrderedDict, defaultdict

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


def test_CNN():
    input = tf.zeros((12, 9, 8))

    cnn_layer = e2edutch.coref_model_new.CNN(tf.shape(input),
                                             test_config['filter_widths'],
                                             test_config['filter_size'])
    output = cnn_layer(input)
    output_shape = tf.shape(output)
    assert tuple(output_shape.numpy()) == (12, 150)


def test_reduce_max():
    x = tf.zeros((4, 2, 4))
    y = e2edutch.coref_model_new.ReduceMax(axis=1)(x)
    assert tuple(tf.shape(y).numpy()) == (4, 4)


def test_LSTM():
    # [num_sentences, max_sentence_length, emb]
    inputs = tf.zeros((2, 6, 10))
    text_len = tf.constant([5, 6])
    lstm = e2edutch.coref_model_new.LSTMContextualize(test_config)
    print('Shape: ', tf.shape(inputs))
    output = lstm(inputs, text_len)
    assert tf.shape(output)[0] == 11  # Total number of words
    assert tf.shape(output)[1] == 2*test_config['contextualization_size']


def test_aggregate_embeddings():
    num_sentences = 2
    max_sentence_length = 6
    emb_size = 10
    max_word_length = 9
    lm_emb_size = test_config['lm_size']
    char_emb_size = test_config['filter_size'] * len(test_config['filter_widths'])
    char_dict = defaultdict(int)
    char_dict[u"<unk>"] = 0
    ds = tf.data.Dataset.from_tensors({'head_word_emb': np.zeros((num_sentences, max_sentence_length, emb_size)),
                                       'context_word_emb': np.zeros((num_sentences, max_sentence_length, emb_size)),
                                       'lm_emb': np.zeros((num_sentences, max_sentence_length, lm_emb_size, 1)),
                                       'char_index': np.zeros((num_sentences, max_sentence_length, max_word_length), dtype='int32')})
    input = list(ds)[0]
    agg_layer = e2edutch.coref_model_new.Aggregate_embedding(char_dict, test_config)
    context_emb, head_emb = agg_layer(input)
    context_emb_shape = tuple(tf.shape(context_emb).numpy())
    head_emb_shape = tuple(tf.shape(head_emb).numpy())
    # [num_sentences, max_sentence_length, emb]
    assert context_emb_shape == (num_sentences, max_sentence_length,
                                 emb_size+lm_emb_size+char_emb_size)
    assert head_emb_shape == (num_sentences, max_sentence_length, emb_size+char_emb_size)

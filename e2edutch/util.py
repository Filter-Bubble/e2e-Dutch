import os
from pathlib import Path
import pyhocon
import errno
import codecs
import collections
import shutil
import logging
import pkg_resources

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

logger = logging.getLogger('e2edutch')


def initialize_from_env(model_name='final', cfg_file=None, model_cfg_file=None):
    '''Read configuration files

    Read configuration files cfg_file and model_cfg_file from provided
    filenames. If none given, use default config files provided by e2edutch:
        cfg/defaults.conf for cfg_file, and
        cfg/models.conf for model_cfg_file


    Configure Tensorflow to use a gpu or cpu based on the environment values of GPU.

    Returns a config dict
    '''
    if "GPU" in os.environ:
        set_gpus(int(os.environ["GPU"]))
    else:
        set_gpus()

    logger.info('Running model: {}'.format(model_name))

    if cfg_file is None:
        cfg_file = pkg_resources.resource_filename(
            "e2edutch", 'cfg/defaults.conf')
    if model_cfg_file is None:
        model_cfg_file = pkg_resources.resource_filename(
            "e2edutch", 'cfg/models.conf')
    config_base = pyhocon.ConfigFactory.parse_file(cfg_file)
    config_model = pyhocon.ConfigFactory.parse_file(model_cfg_file)[model_name]
    config = pyhocon.ConfigTree.merge_configs(config_model, config_base)

    # Override datapath from environment, if set
    if os.environ.get('E2E_HOME', None) is not None:
        config['datapath'] = os.environ['E2E_HOME']

    # Finally, provide fallback for datapath
    if config.get('datapath', None) is None:
        config['datapath'] = Path(__file__).parent / "data"

    config['log_root'] = config['datapath']
    config['log_dir'] = model_name

    mkdirs(os.path.join(config['log_root'], config['log_dir']))

    logger.debug(pyhocon.HOCONConverter.convert(config, 'hocon'))
    return config


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source + ext, target + ext)


def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(
        tag=k, simple_value=v) for k, v in value_dict.items()])


def create_example(text, doc_key='example'):
    import nltk
    nltk.download("punkt")
    from nltk.tokenize import sent_tokenize, word_tokenize
    raw_sentences = sent_tokenize(text)
    sentences = [word_tokenize(s, language='dutch') for s in raw_sentences]
    return {
        "doc_key": doc_key,
        "clusters": [],
        "sentences": sentences
    }


def flatten(l):
    return [item for sublist in l for item in sublist]


def set_gpus(*gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    logger.info("Setting CUDA_VISIBLE_DEVICES to: {}".format(
        os.environ["CUDA_VISIBLE_DEVICES"]))
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with codecs.open(char_vocab_path, encoding="utf-8") as f:
        vocab.extend(l.strip() for l in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


def projection(inputs, output_size, initializer=None):
    return ffnn(inputs, 0, -1, output_size, dropout=None,
                output_weights_initializer=initializer)


def highway(inputs, num_layers, dropout):
    for i in range(num_layers):
        with tf.variable_scope("highway_{}".format(i)):
            j, f = tf.split(projection(inputs, 2 * shape(inputs, -1)), 2, -1)
            f = tf.sigmoid(f)
            j = tf.nn.relu(j)
            if dropout is not None:
                j = tf.nn.dropout(j, dropout)
            inputs = f * j + (1 - f) * inputs
    return inputs


def shape(x, dim):
    return x.get_shape()[dim].value or tf.shape(x)[dim]


def ffnn(inputs, num_hidden_layers, hidden_size, output_size,
         dropout, output_weights_initializer=None):
    if len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(
            len(inputs.get_shape())))

    if len(inputs.get_shape()) == 3:
        batch_size = shape(inputs, 0)
        seqlen = shape(inputs, 1)
        emb_size = shape(inputs, 2)
        current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.get_variable("hidden_weights_{}".format(i),
                                         [shape(current_inputs, 1),
                                          hidden_size])
        hidden_bias = tf.get_variable(
            "hidden_bias_{}".format(i), [hidden_size])
        current_outputs = tf.nn.relu(tf.nn.xw_plus_b(
            current_inputs, hidden_weights, hidden_bias))

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, dropout)
        current_inputs = current_outputs

    output_weights = tf.get_variable("output_weights", [shape(
        current_inputs, 1), output_size],
        initializer=output_weights_initializer)
    output_bias = tf.get_variable("output_bias", [output_size])
    outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
    return outputs


def cnn(inputs, filter_sizes, num_filters):
    # num_words = shape(inputs, 0)
    # num_chars = shape(inputs, 1)
    input_size = shape(inputs, 2)
    outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv_{}".format(i)):
            w = tf.get_variable("w", [filter_size, input_size, num_filters])
            b = tf.get_variable("b", [num_filters])
        # [num_words, num_chars - filter_size, num_filters]
        conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID")
        # [num_words, num_chars - filter_size, num_filters]
        h = tf.nn.relu(tf.nn.bias_add(conv, b))
        pooled = tf.reduce_max(h, 1)  # [num_words, num_filters]
        outputs.append(pooled)
    # [num_words, num_filters * len(filter_sizes)]
    return tf.concat(outputs, 1)


def batch_gather(emb, indices):
    batch_size = shape(emb, 0)
    seqlen = shape(emb, 1)
    if len(emb.get_shape()) > 2:
        emb_size = shape(emb, 2)
    else:
        emb_size = 1
    # [batch_size * seqlen, emb]
    flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])
    offset = tf.expand_dims(tf.range(batch_size)
                            * seqlen, 1)  # [batch_size, 1]
    # [batch_size, num_indices, emb]
    gathered = tf.gather(flattened_emb, indices + offset)
    if len(emb.get_shape()) == 2:
        gathered = tf.squeeze(gathered, 2)  # [batch_size, num_indices]
    return gathered


class RetrievalEvaluator(object):
    def __init__(self):
        self._num_correct = 0
        self._num_gold = 0
        self._num_predicted = 0

    def update(self, gold_set, predicted_set):
        self._num_correct += len(gold_set & predicted_set)
        self._num_gold += len(gold_set)
        self._num_predicted += len(predicted_set)

    def recall(self):
        return maybe_divide(self._num_correct, self._num_gold)

    def precision(self):
        return maybe_divide(self._num_correct, self._num_predicted)

    def metrics(self):
        recall = self.recall()
        precision = self.precision()
        f1 = maybe_divide(2 * recall * precision, precision + recall)
        return recall, precision, f1


class EmbeddingDictionary(object):
    def __init__(self, info, datapath='', normalize=True, maybe_cache=None):
        self._size = info["size"]
        self._normalize = normalize
        self._path = os.path.join(datapath, info["path"])
        self._cased = info["cased"] if "cased" in info else True
        if maybe_cache is not None and maybe_cache._path == self._path:
            assert self._size == maybe_cache._size
            self._embeddings = maybe_cache._embeddings
        else:
            self._embeddings = self.load_embedding_dict(self._path)

    @property
    def size(self):
        return self._size

    def load_embedding_dict(self, path):
        logger.info("Loading word embeddings from {}...".format(path))
        default_embedding = np.zeros(self.size)
        embedding_dict = collections.defaultdict(lambda: default_embedding)
        if len(path) > 0:
            vocab_size = None
            with open(path) as f:
                for i, line in enumerate(f.readlines()):
                    word_end = line.find(" ")
                    word = line[:word_end]
                    embedding = np.fromstring(
                        line[word_end + 1:], np.float32, sep=" ")
                    assert len(embedding) == self.size
                    embedding_dict[word] = embedding
            if vocab_size is not None:
                assert vocab_size == len(embedding_dict)
            logger.info("Done loading word embeddings.")
        return embedding_dict

    def __getitem__(self, key):
        if not self._cased:
            key = key.lower()
        embedding = self._embeddings[key]
        if self._normalize:
            embedding = self.normalize(embedding)
        return embedding

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        else:
            return v


class CustomLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, batch_size, dropout):
        self._num_units = num_units
        self._dropout = dropout
        self._dropout_mask = tf.nn.dropout(
            tf.ones([batch_size, self.output_size]), dropout)
        self._initializer = self._block_orthonormal_initializer(
            [self.output_size] * 3)
        initial_cell_state = tf.get_variable(
            "lstm_initial_cell_state", [1, self.output_size])
        initial_hidden_state = tf.get_variable(
            "lstm_initial_hidden_state", [1, self.output_size])
        self._initial_state = tf.nn.rnn_cell.LSTMStateTuple(
            initial_cell_state, initial_hidden_state)

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(
            self.output_size, self.output_size)

    @property
    def output_size(self):
        return self._num_units

    @property
    def initial_state(self):
        return self._initial_state

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # CustomLSTMCell
            c, h = state
            h *= self._dropout_mask
            concat = projection(
                tf.concat([inputs, h], 1), 3 * self.output_size,
                initializer=self._initializer)
            i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
            i = tf.sigmoid(i)
            new_c = (1 - i) * c + i * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

    def _orthonormal_initializer(self, scale=1.0):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
            M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
            Q1, R1 = np.linalg.qr(M1)
            Q2, R2 = np.linalg.qr(M2)
            Q1 = Q1 * np.sign(np.diag(R1))
            Q2 = Q2 * np.sign(np.diag(R2))
            n_min = min(shape[0], shape[1])
            params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            return params
        return _initializer

    def _block_orthonormal_initializer(self, output_sizes):
        def _initializer(shape, dtype=np.float32, partition_info=None):
            assert len(shape) == 2
            assert sum(output_sizes) == shape[1]
            initializer = self._orthonormal_initializer()
            params = np.concatenate(
                [initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
            return params
        return _initializer

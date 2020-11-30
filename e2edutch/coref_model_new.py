import tensorflow as tf

# from . import bert
from . import util
# from . import coref_ops
# from . import conll
# from . import metrics


class ReduceMax(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(ReduceMax, self).__init__(**kwargs)
        self.axis = axis

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([j for i, j in enumerate(input_shape)
                               if j != self.axis])

    def call(self, inputs):
        return tf.reduce_max(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(ReduceMax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CNN(tf.keras.layers.Layer):
    def __init__(self, input_shape, filter_sizes, num_filters):
        super(CNN, self).__init__()
        input_layer = tf.keras.Input(shape=input_shape)
        conv_layers = []
        for i, filter_size in enumerate(filter_sizes):
            conv_layer = tf.keras.layers.Conv1D(num_filters, filter_size,
                                                strides=1, padding='valid',
                                                activation='relu')(input_layer)
            pool_layer = ReduceMax(axis=-2)(conv_layer)
            conv_layers.append(pool_layer)

        output_layer = tf.keras.layers.Concatenate()(conv_layers)
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    def call(self, x):
        flattened_x = tf.reshape(x, (1, *tf.shape(x)))
        output = self.model(flattened_x)
        output = tf.reshape(output, tf.shape(output)[1:])
        return output


class Aggregate_embedding(tf.keras.layers.Layer):
    def __init__(self, char_dict, config):
        super(Aggregate_embedding, self).__init__()
        self.config = config
        self.char_dict = char_dict

        # tensorflow trainable variables:
        self.lm_scores = tf.Variable(tf.constant(0.0, shape=[config['lm_layers']]),
                                     name='lm_scores')
        self.lm_scaling = tf.Variable(tf.constant(1.0, shape=[]),
                                      name='lm_scaling')
        if self.config['char_embedding_size'] > 0:
            self.char_embeddings = tf.Variable(tf.zeros(shape=[len(self.char_dict),
                                                               self.config['char_embedding_size']]),
                                               name='char_embeddings')
        else:
            self.char_embeddings = None

        self.cnn = CNN((None, None, self.config['char_embedding_size']),
                       self.config['filter_widths'],
                       self.config['filter_size'])

    def call(self, inputs):
        lm_emb = inputs['lm_emb']
        context_word_emb = inputs['context_word_emb']
        head_word_emb = inputs['head_word_emb']

        num_sentences = tf.shape(input=context_word_emb)[0]
        max_sentence_length = tf.shape(input=context_word_emb)[1]

        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]

        if self.config['char_embedding_size'] > 0:
            # [num_sentences, max_sentence_length, max_word_length, emb]
            char_emb = tf.gather(self.char_embeddings, inputs['char_index'])

            # [num_sentences * max_sentence_length, max_word_length, emb]
            flattened_char_emb = tf.reshape(char_emb,
                                            [num_sentences
                                             * max_sentence_length,
                                             char_emb.shape[2],
                                             char_emb.shape[3]])

            # [num_sentences * max_sentence_length, emb]
            flattened_aggregated_char_emb = self.cnn(flattened_char_emb)

            # [num_sentences, max_sentence_length, emb]
            aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb,
                                             [num_sentences,
                                              max_sentence_length,
                                              util.shape(
                                                  flattened_aggregated_char_emb,
                                                  1)])
            context_emb_list.append(aggregated_char_emb)
            head_emb_list.append(aggregated_char_emb)

        lm_emb_size = util.shape(lm_emb, 2)
        lm_num_layers = util.shape(lm_emb, 3)
        lm_weights = tf.nn.softmax(self.lm_scores)
        flattened_lm_emb = tf.reshape(
            lm_emb, [num_sentences
                     * max_sentence_length
                     * lm_emb_size,
                     lm_num_layers])

        # [num_sentences * max_sentence_length * emb, 1]
        flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb,
                                                tf.expand_dims(
                                                    lm_weights, 1))
        aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [
            num_sentences, max_sentence_length,
            lm_emb_size])
        aggregated_lm_emb *= self.lm_scaling
        context_emb_list.append(aggregated_lm_emb)

        # [num_sentences, max_sentence_length, emb]
        context_emb = tf.concat(context_emb_list, 2)
        # [num_sentences, max_sentence_length, emb]
        head_emb = tf.concat(head_emb_list, 2)

        return context_emb, head_emb


class CorefModel(tf.keras.Model):
    def __init__(self, config):
        super(CorefModel, self).__init__()
        self.config = config

    def call(self, inputs):
        outputs = inputs
        return outputs

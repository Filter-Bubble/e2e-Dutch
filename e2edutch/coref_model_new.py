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


class Aggregate_embedding(tf.keras.Model):
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


class LSTMContextualize(tf.keras.layers.Layer):
    def __init__(self, config):
        super(LSTMContextualize, self).__init__()
        self.config = config
        self.lstm_dropout = config['lstm_dropout_rate']
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=self.config["contextualization_size"],
                                 dropout=self.lstm_dropout,
                                 return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(self.lstm_dropout)

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(input=emb)[0]
        max_sentence_length = tf.shape(input=emb)[1]
        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(
                emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(
                emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        mask_reshaped = tf.reshape(
            text_len_mask, [num_sentences * max_sentence_length])
        return tf.boolean_mask(tensor=flattened_emb, mask=mask_reshaped)

    def call(self, inputs, text_len):
        max_sentence_length = tf.shape(inputs)[-2]
        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)
        # [num_sentences, max_sentence_length, emb]
        text_outputs = self.lstm(inputs)
        # [num_sentences, max_sentence_length, emb]
        text_outputs = self.dropout(text_outputs)
        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)


class CorefModel(tf.keras.Model):
    def __init__(self, config, char_dict):
        super(CorefModel, self).__init__()
        self.config = config
        self.char_dict = char_dict
        self.aggregate_embedding_layer = Aggregate_embedding(self.char_dict, self.config)
        self.dropout_layer = tf.keras.layers.Dropout(config['lexical_dropout_rate'])

    def call(self, inputs):
        context_emb, head_emb = self.aggregate_embedding_layer(inputs)
        context_emb = self.dropout_layer(context_emb, inputs['is_training'])
        head_emb = self.dropout_layer(head_emb, inputs['is_training'])
        outputs = (context_emb, head_emb)
        return outputs

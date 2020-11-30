import os
import tensorflow as tf
import numpy as np
import h5py
import random

from . import bert
from . import util


class DataTransformer(object):
    def __init__(self, config):
        self.config = config
        self.context_embeddings = util.EmbeddingDictionary(
            config["context_embeddings"], config['datapath'])
        self.head_embeddings = util.EmbeddingDictionary(
            config["context_embeddings"], config['datapath'],
            maybe_cache=self.context_embeddings)
        self.char_embedding_size = config["char_embedding_size"]
        self.char_dict = util.load_char_dict(
            os.path.join(config['datapath'],
                         config["char_vocab_path"]))
        self.max_span_width = config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        if config["lm_path"]:
            self.lm_file = h5py.File(os.path.join(config['datapath'],
                                                  self.config["lm_path"]), "r")
        else:
            self.lm_file = None
        if config["lm_model_name"]:
            self.bert_tokenizer, self.bert_model = bert.load_bert(
                self.config["lm_model_name"])
        else:
            self.bert_tokenizer = None
            self.bert_model = None
        self.lm_layers = self.config["lm_layers"]
        self.lm_size = self.config["lm_size"]

    def load_lm_embeddings(self, doc_key, sentences):
        if self.lm_file is None and self.bert_model is None:
            # No LM specified
            return np.zeros([0, 0, self.lm_size, self.lm_layers])
        elif self.lm_file is None:
            # No cache file, encode on the fly
            lm_emb = bert.encode_sentences(
                sentences, self.bert_tokenizer, self.bert_model)
            return lm_emb
        file_key = doc_key.replace("/", ":")
        group = self.lm_file.get(file_key, None)
        if group is None and self.bert_model is not None:
            # Document not cached, encode on the fly
            lm_emb = bert.encode_sentences(
                sentences, self.bert_tokenizer, self.bert_model)
        elif group is not None:
            # Load encoding from cache file
            num_sentences = len(list(group.keys()))
            sentences = [group[str(i)][...] for i in range(num_sentences)]
            lm_emb = np.zeros([num_sentences, max(s.shape[0]
                                                  for s in sentences),
                               self.lm_size, self.lm_layers])
            for i, s in enumerate(sentences):
                lm_emb[i, :s.shape[0], :, :] = s
        return lm_emb

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_span_labels(self, tuples, label_dict):
        if len(tuples) > 0:
            starts, ends, labels = zip(*tuples)
        else:
            starts, ends, labels = [], [], []
        return (np.array(starts),
                np.array(ends),
                np.array([label_dict[c] for c in labels]))

    def tensorize_example(self, example, is_training):
        example_tensors = {}
        clusters = example["clusters"]

        gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
        example_tensors['cluster_ids'] = cluster_ids
        sentences = example["sentences"]

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s)
                                  for s in sentences),
                              max(self.config["filter_widths"]))
        example_tensors['text_len'] = np.array([len(s) for s in sentences])
        tokens = [[""] * max_sentence_length for _ in sentences]
        context_word_emb = np.zeros(
            [len(sentences),
             max_sentence_length,
             self.context_embeddings.size])
        head_word_emb = np.zeros(
            [len(sentences),
             max_sentence_length,
             self.head_embeddings.size])
        char_index = np.zeros(
            [len(sentences), max_sentence_length, max_word_length], dtype='int32')
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                tokens[i][j] = word
                context_word_emb[i, j] = self.context_embeddings[word]
                head_word_emb[i, j] = self.head_embeddings[word]
                char_index[i, j, :len(word)] = [self.char_dict[c]
                                                for c in word]
        example_tensors['tokens'] = np.array(tokens)
        example_tensors['head_word_emb'] = head_word_emb
        example_tensors['context_word_emb'] = context_word_emb
        example_tensors['char_index'] = char_index
        example_tensors['genre'] = self.genres[example.get('genre', 'all')]

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
        example_tensors['gold_starts'] = gold_starts
        example_tensors['gold_ends'] = gold_ends

        example_tensors['lm_emb'] = self.load_lm_embeddings(
            example["doc_key"], example["sentences"])

        example_tensors['is_training'] = is_training

        if is_training and len(
                sentences) > self.config["max_training_sentences"]:
            example = self.truncate_example(example_tensors)
        return tf.data.Dataset.from_tensors(example_tensors)

    def truncate_example(self, example_tensors):
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = example_tensors['context_word_emb'].shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(
            0, num_sentences - max_training_sentences)
        word_offset = example_tensors['text_len'][:sentence_offset].sum()
        num_words = example_tensors['text_len'][sentence_offset:sentence_offset
                                                + max_training_sentences].sum()
        tokens = example_tensors['tokens'][sentence_offset:sentence_offset
                                           + max_training_sentences, :]
        context_word_emb = example_tensors['context_word_emb'][sentence_offset:sentence_offset
                                                               + max_training_sentences, :, :]
        head_word_emb = example_tensors['head_word_emb'][sentence_offset:sentence_offset
                                                         + max_training_sentences, :, :]
        lm_emb = example_tensors['lm_emb'][sentence_offset:sentence_offset
                                           + max_training_sentences, :, :, :]
        char_index = example_tensors['char_index'][sentence_offset:sentence_offset
                                                   + max_training_sentences, :, :]
        text_len = example_tensors['text_len'][sentence_offset:sentence_offset
                                               + max_training_sentences]

        gold_spans = np.logical_and(
            example_tensors['gold_ends'] >= word_offset,
            example_tensors['gold_starts'] < word_offset + num_words)
        gold_starts = example_tensors['gold_starts'][gold_spans] - word_offset
        gold_ends = example_tensors['gold_ends'][gold_spans] - word_offset
        cluster_ids = example_tensors['cluster_ids'][gold_spans]

        return {'tokens': tokens, 'context_word_emb': context_word_emb,
                'head_word_emb': head_word_emb,
                'lm_emb': lm_emb,
                'char_index': char_index,
                'text_len': text_len,
                'genre': example_tensors['genre'],
                'is_training': example_tensors['is_training'],
                'gold_starts': gold_starts,
                'gold_ends': gold_ends,
                'cluster_ids': cluster_ids}

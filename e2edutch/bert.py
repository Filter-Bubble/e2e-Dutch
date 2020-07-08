import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, BertForPreTraining, BertConfig

import logging


def load_bert(model_name):
    if model_name == 'robbert':
        tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robBERT-base")
        model = RobertaModel.from_pretrained("pdelobelle/robBERT-base")
    elif model_name == 'bertje':
        tokenizer = BertTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased")
        model = BertModel.from_pretrained("wietsedv/bert-base-dutch-cased")
    elif model_name == 'bert-nl':
        tokenizer = BertTokenizer.from_pretrained("data/bert-nl")
        config = BertConfig.from_json_file("data/bert-nl/config.json")
        model = BertForPreTraining(config).bert
    else:
        raise ValueError('invalid model name')
    model.eval()
    return tokenizer, model


def encode_sentences(sentences, tokenizer, model):
    # Use BERT tokenizer
    sentences_tokenized = [
        [tokenizer.tokenize(word) for word in sentence] for sentence in sentences]
    sentences_tokenized_flat = [
        [tok for word in sentence for tok in word] for sentence in sentences_tokenized]
    indices_flat = [[i for i, word in enumerate(
        sentence) for tok in word] for sentence in sentences_tokenized]

    max_nrtokens = max(len(s) for s in sentences_tokenized_flat)
    indexed_tokens = np.zeros((len(sentences), max_nrtokens), dtype=int)
    for i, sent in enumerate(sentences_tokenized_flat):
        idx = tokenizer.convert_tokens_to_ids(sent)
        indexed_tokens[i, :len(idx)] = np.array(idx)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor(indexed_tokens)
    with torch.no_grad():
        # torch tensor of shape
        # (nr_sentences, sequence_length, hidden_size=768
        bert_output, _ = model(tokens_tensor)

    # Add up tensors for subtokens coming from same word
    max_sentence_length = max(len(s) for s in sentences)
    bert_final = torch.tensor(np.zeros((bert_output.shape[0],
                                        max_sentence_length,
                                        bert_output.shape[2])))
    for sent_id in range(len(sentences)):
        for tok_id, word_id in enumerate(indices_flat[sent_id]):
            bert_final[sent_id, word_id, :] += bert_output[sent_id, tok_id, :]
    bert_final = np.array(bert_final)
    # Add extra axis
    bert_final = np.expand_dims(bert_final, axis=3)
    return bert_final

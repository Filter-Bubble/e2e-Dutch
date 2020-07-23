import re
import os
import sys
import json
import tempfile
import subprocess
import collections
import logging

from e2edutch import util
from e2edutch import conll


class DocumentState(object):
    def __init__(self):
        self.doc_key = None
        self.text = []
        self.sentences = []
        self.constituents = {}
        self.const_stack = []
        self.ner = {}
        self.ner_stack = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)

    def assert_empty(self):
        assert self.doc_key is None
        assert len(self.text) == 0
        assert len(self.sentences) == 0
        assert len(self.constituents) == 0
        assert len(self.const_stack) == 0
        assert len(self.ner) == 0
        assert len(self.ner_stack) == 0
        assert len(self.coref_stacks) == 0
        assert len(self.clusters) == 0

    def assert_finalizable(self):
        assert self.doc_key is not None
        assert len(self.text) == 0
        assert len(self.sentences) > 0
        #assert len(self.constituents) > 0
        assert len(self.const_stack) == 0
        assert len(self.ner_stack) == 0
        assert all(len(s) == 0 for s in self.coref_stacks.values())

    def span_dict_to_list(self, span_dict):
        return [(s, e, l) for (s, e), l in span_dict.items()]

    def finalize(self):
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often.)")
                print(self.doc_key, m)
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        merged_clusters = [list(c) for c in merged_clusters]
        all_mentions = util.flatten(merged_clusters)
        assert len(all_mentions) == len(set(all_mentions))

        return {
            "doc_key": self.doc_key,
            "sentences": self.sentences,
            "clusters": merged_clusters
        }


def normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def handle_bit(word_index, bit, stack, spans):
    asterisk_idx = bit.find("*")
    if asterisk_idx >= 0:
        open_parens = bit[:asterisk_idx]
        close_parens = bit[asterisk_idx + 1:]
    else:
        open_parens = bit[:-1]
        close_parens = bit[-1]

    current_idx = open_parens.find("(")
    while current_idx >= 0:
        next_idx = open_parens.find("(", current_idx + 1)
        if next_idx >= 0:
            label = open_parens[current_idx + 1:next_idx]
        else:
            label = open_parens[current_idx + 1:]
        stack.append((word_index, label))
        current_idx = next_idx

    for c in close_parens:
        assert c == ")"
        open_index, label = stack.pop()
        current_span = (open_index, word_index)
        """
    if current_span in spans:
      spans[current_span] += "_" + label
    else:
      spans[current_span] = label
    """
        spans[current_span] = label


def handle_line(line, document_state, labels, stats, word_col):
    begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
    if begin_document_match:
        document_state.assert_empty()
        document_state.doc_key = conll.get_doc_key(*begin_document_match.groups())
        return None
    elif line.startswith("#end document"):
        if len(document_state.text) > 0:  # no newline before end document
            stats["max_sent_len"] = max(
                len(document_state.text), stats["max_sent_len"])
            stats["num_sents"] += 1
            document_state.sentences.append(tuple(document_state.text))
            del document_state.text[:]
        document_state.assert_finalizable()
        finalized_state = document_state.finalize()
        stats["num_clusters"] += len(finalized_state["clusters"])
        stats["num_mentions"] += sum(len(c)
                                     for c in finalized_state["clusters"])
        # labels["const_labels"].update(
        #     l for _, _, l in finalized_state["constituents"])
        # labels["ner"].update(l for _, _, l in finalized_state["ner"])
        return finalized_state
    else:
        row = line.split()
        if len(row) == 0 and len(document_state.text) > 0:
            stats["max_sent_len"] = max(
                len(document_state.text), stats["max_sent_len"])
            stats["num_sents"] += 1
            document_state.sentences.append(tuple(document_state.text))
            del document_state.text[:]
            return None
        elif len(row) == 0 and len(document_state.text) == 0:
            return None
        assert len(row) >= 4


        word = normalize_word(row[word_col])
        coref = row[-1]

        word_index = len(document_state.text) + sum(len(s)
                                                    for s in document_state.sentences)
        document_state.text.append(word)

        if coref != "-":
            for segment in coref.split("|"):
                if segment[0] == "(":
                    if segment[-1] == ")":
                        cluster_id = int(segment[1:-1])
                        document_state.clusters[cluster_id].append(
                            (word_index, word_index))
                    else:
                        cluster_id = int(segment[1:])
                        document_state.coref_stacks[cluster_id].append(
                            word_index)
                else:
                    cluster_id = int(segment[:-1])
                    start = document_state.coref_stacks[cluster_id].pop()
                    document_state.clusters[cluster_id].append(
                        (start, word_index))
        return None


def minimize_partition(input_path, labels, stats, word_col):
    with open(input_path, "r") as input_file:
        document_state = DocumentState()
        for line in input_file.readlines():
            document = handle_line(line, document_state,
                                   labels, stats, word_col)
            if document is not None:
                yield document
                document_state = DocumentState()


def minimize_partition_file(input_path, labels, stats, word_col):
    output_path = "{}.jsonlines".format(os.path.splitext(input_path)[0])
    count = 0
    logging.info("Minimizing {}".format(input_path))

    with open(output_path, "w") as output_file:
        for document in minimize_partition(input_path, labels, stats, word_col):
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    logging.info("Wrote {} documents to {}".format(count, output_path))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(
            "Usage: {} input file".format(sys.argv[0]))
    input_path = sys.argv[1]
    word_col = 2
    labels = collections.defaultdict(set)
    stats = collections.defaultdict(int)
    minimize_partition_file(input_path, labels, stats, word_col)
    for k, v in labels.items():
        print("{} = [{}]".format(k, ", ".join(
            "\"{}\"".format(label) for label in v)))
    for k, v in stats.items():
        print("{} = {}".format(k, v))

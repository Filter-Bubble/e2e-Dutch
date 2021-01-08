import re
import tempfile
import subprocess
import operator
import collections

BEGIN_DOCUMENT_REGEX = re.compile(
    r"#begin document \(?([^\);]*)\)?;?(?: part (\d+))?")
COREF_RESULTS_REGEX = re.compile(
    r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)


def get_doc_key(doc_id, part=None):
    if part is None:
        return doc_id
    else:
        return '{}.p.{}'.format(doc_id, part)


def get_reverse_doc_key(doc_key):
    segments = doc_key.split('.p.')
    if len(segments) > 1:
        part = segments[-1]
        doc_id = '.p.'.join(segments[:-1])
    else:
        doc_id = doc_key
        part = None
    return doc_id, part


def get_prediction_map(predictions):
    prediction_map = {}
    for doc_key, clusters in predictions.items():
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k, v in start_map.items():
            start_map[k] = [cluster_id for cluster_id, end in sorted(
                v, key=operator.itemgetter(1), reverse=True)]
        for k, v in end_map.items():
            end_map[k] = [cluster_id for cluster_id, start in sorted(
                v, key=operator.itemgetter(1), reverse=True)]
        prediction_map[doc_key] = (start_map, end_map, word_map)
    return prediction_map


def clusters_to_brackets(sentences, predictions):
    prediction_map = get_prediction_map({'': predictions})
    start_map, end_map, word_map = prediction_map['']
    word_index = 0
    brackets_list = []
    for sent in sentences:
        sent_brackets_list = []
        for i, word in enumerate(sent):
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))
            coref = '-' if len(coref_list) == 0 else "|".join(coref_list)
            sent_brackets_list.append(coref)
            word_index += 1
        brackets_list.append(sent_brackets_list)
    return brackets_list


def output_conll(output_file, sentences, predictions):
    """
    Output the tokens and coreferences in CONLL-2012 format

    Args:
        output_file (File or IOBase): File to write the CONLL to
        sentences (dict): keys are the doc_keys, values are the sentences of
                          that doc
        predictions (dict): keys are the doc_keys, values are the predicted
                            clusters of that doc
    """
    for doc_key in sentences:
        brackets = clusters_to_brackets(sentences[doc_key], predictions[doc_key])
        doc_id, part = get_reverse_doc_key(doc_key)
        if part is None:
            output_file.write("#begin document ({});\n\n".format(doc_id))
        else:
            output_file.write(
                "#begin document ({}); part {}\n\n".format(
                    doc_id, part))
        for sent, brack_sent in zip(sentences[doc_key], brackets):
            for i, word in enumerate(sent):
                coref = brack_sent[i]
                line = '\t'.join([doc_id, str(i), word, coref])
                output_file.write(line + '\n')
            output_file.write('\n')
        output_file.write('#end document\n')


def output_conll_align(input_file, output_file, predictions):
    prediction_map = get_prediction_map(predictions)

    word_index = 0
    for line in input_file.readlines():
        row = line.split()
        if len(row) == 0:
            output_file.write("\n")
        elif row[0].startswith("#"):
            begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
            if begin_match:
                doc_key = get_doc_key(*begin_match.groups())
                start_map, end_map, word_map = prediction_map[doc_key]
                word_index = 0
            output_file.write(line)
            output_file.write("\n")
        else:
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))

            if len(coref_list) == 0:
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            output_file.write("   ".join(row))
            output_file.write("\n")
            word_index += 1


def official_conll_eval(gold_path, predicted_path,
                        metric, official_stdout=False):
    cmd = ["conll-2012/scorer/v8.01/scorer.pl",
           metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        print(stderr)

    if official_stdout:
        print("Official result for {}".format(metric))
        print(stdout)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return {"r": recall, "p": precision, "f": f1}


def evaluate_conll(gold_path, predictions, official_stdout=False):
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as pred_file:
        with open(gold_path, "r") as gold_file:
            output_conll_align(gold_file, pred_file, predictions)
        print("Predicted conll file: {}".format(pred_file.name))
    return {m: official_conll_eval(
        gold_file.name, pred_file.name, m, official_stdout)
        for m in ("muc", "bcub", "ceafe")}

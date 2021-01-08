import e2edutch.conll
import os
__here__ = os.path.dirname(os.path.realpath(__file__))


def test_get_doc_key():
    doc_key = e2edutch.conll.get_doc_key('test', '1')
    assert doc_key == 'test.p.1'


def test_get_doc_key_nopart():
    doc_key = e2edutch.conll.get_doc_key('test')
    assert doc_key == 'test'


def test_get_prediction_map():
    predictions = {'doc1': [[(0, 1)]]}
    prediction_map = e2edutch.conll.get_prediction_map(predictions)
    assert len(prediction_map) == 1
    assert len(prediction_map['doc1']) == 3
    start_map, end_map, word_map = prediction_map['doc1']
    assert start_map[0] == [0]
    assert end_map[1] == [0]
    assert word_map == {}


def test_predictions_to_brackets():
    sentences = [['Een', 'zin', '.'],
                 ['Nog', 'een', 'zin']]
    predictions = [[(1, 1)], [(4, 5)]]
    brackets = e2edutch.conll.clusters_to_brackets(sentences, predictions)
    assert len(brackets) == 2
    assert len(brackets[0]) == 3
    assert brackets[0][1] == '(0)'
    assert brackets[1][1] == '(1'
    assert brackets[1][2] == '1)'


def test_output_conll():
    output_file = '/tmp/tmp.conll'
    sentences = {'doc1.p.1': [['Dit', 'is', 'een', 'test', '.']]}
    predictions = {'doc1.p.1': [[(0, 0), (2, 3)]]}
    with open(output_file, 'w') as fout:
        e2edutch.conll.output_conll(fout, sentences, predictions)
    assert os.path.exists(output_file)
    content = open(output_file).readlines()
    nonempty = [line for line in content if line.strip() != '']
    assert len(nonempty) == 7
    assert content[0].strip() == '#begin document (doc1); part 1'


def test_output_conll_align():
    input_file = os.path.join(__here__, 'data', 'test.conll')
    output_file = '/tmp/tmp.conll'
    predictions = {'doc1.p.1': [[(0, 0), (2, 3)]]}
    with open(input_file) as fin:
        with open(output_file, 'w') as fout:
            e2edutch.conll.output_conll_align(fin, fout, predictions)
    content = open(output_file).readlines()
    nonempty = [line for line in content if line.strip() != '']
    assert len(nonempty) == 7
    assert content[0].strip() == '#begin document (doc1); part 1'

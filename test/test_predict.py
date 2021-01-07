import e2edutch.predict
from collections import OrderedDict
import tensorflow.compat.v1 as tf


def test_predict():
    example = {
        'doc_key': 'test',
        'sentences': [['Een', 'zin', '.']]
    }
    predictor = e2edutch.predict.Predictor(model_name='final')
    clusters = predictor.predict(example)
    predictor.end_session()
    assert(len(clusters) >= 0)

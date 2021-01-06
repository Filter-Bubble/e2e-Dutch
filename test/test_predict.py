import e2edutch.predict
from collections import OrderedDict


def test_predict():
    example = {
        'doc_key': 'test',
        'sentences': [['Een', 'zin', '.']]
    }
    predictor = e2edutch.predict.Predictor(model_name='best')
    clusters = predictor.predict(example)
    predictor.end_session()
    assert(len(clusters) >= 0)

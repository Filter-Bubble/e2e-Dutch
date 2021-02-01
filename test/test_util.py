from e2edutch import util


def test_create_example():
    text = 'Dit is een zin.'
    example = util.create_example(text, doc_key='test')
    assert example['doc_key'] == 'test'
    assert len(example['sentences']) > 0
    assert len(example['sentences'][0]) > 0

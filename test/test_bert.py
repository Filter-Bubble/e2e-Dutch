import e2edutch.bert

def test_encode_sentences():
    sentences = ['Dit is een zin .'.split(' '), 'Het is slechts een voorbeeld .'.split(' ')]
    tokenizer, model = e2edutch.bert.load_bert('bertje')
    emb = e2edutch.bert.encode_sentences(sentences, tokenizer, model)
    assert emb.shape == (2, 6, 768, 1)

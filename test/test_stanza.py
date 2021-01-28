import stanza
import e2edutch.stanza
import tensorflow.compat.v1 as tf


def test_processor():
    nlp = stanza.Pipeline(lang='nl', processors='tokenize,coref')
    text = 'Dit is een tekst.'
    doc = nlp(text)
    # TODO: asserts about the doc having corefs

from . import __version__
import logging

import KafNafParserPy
from KafNafParserPy import KafNafParser
from lxml.etree import XMLSyntaxError
import itertools
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

logger = logging.getLogger('e2edutch')
this_name = 'Coreference resolution based on e2e model'


def get_naf(input_filename):
    try:
        naf = KafNafParser(input_filename)
    except XMLSyntaxError:
        with open(input_filename) as input_file:
            input = input_file.read()
        if "<NAF" in input and "</NAF>" in input:
            # I'm guessing this should be a NAF file but something is wrong
            logger.exception("Error parsing NAF file")
            raise
        naf = KafNafParser(type="NAF")
        naf.set_version("3.0")
        naf.set_language("nl")
        naf.lang = "nl"
        naf.raw = input
        naf.set_raw(naf.raw)
    return naf


def get_naf_from_sentences(sentences):
    naf_obj = KafNafParser(type="NAF")
    naf_obj.set_version("3.0")
    naf_obj.set_language("nl")
    naf_obj.lang = "nl"
    naf_obj.raw = '\n'.join([' '.join(s) for s in sentences])
    naf_obj.set_raw(naf_obj.raw)
    # Create text layer
    wcount = 1
    offsets = {}
    txt = naf_obj.get_raw()
    token_ids = []
    for sid, sentence in enumerate(sentences):
        token_ids_sub = []
        for token in sentence:
            token_obj = KafNafParserPy.Cwf(type=naf_obj.get_type())
            token_id = 'w{}'.format(wcount)
            token_length = len(token)
            offsets[wcount] = txt.find(token, offsets.get(wcount - 1, 0))
            token_obj.set_id(token_id)
            token_obj.set_length(str(token_length))
            # token_obj.set_offset(str(offset)) # Is this correct????
            token_obj.set_para('1')
            token_obj.set_sent(str(sid + 1))
            token_obj.set_text(token)
            token_obj.set_offset(str(offsets[wcount]))
            token_ids_sub.append(token_id)
            wcount += 1
            naf_obj.add_wf(token_obj)
        token_ids.append(token_ids_sub)
    # Create term layers
    term_ids = []
    count_terms = 0
    for sid, (sentence, token_ids_sub) in enumerate(zip(sentences, token_ids)):
        term_ids_sub = []
        logger.info('Creating the term layer...')
        for num_token, (token, token_id) in enumerate(
                zip(sentence, token_ids_sub)):
            new_term_id = 't_' + str(count_terms)
            count_terms += 1
            term_ids_sub.append(new_term_id)
            term_obj = KafNafParserPy.Cterm(type=naf_obj.get_type())
            term_obj.set_id(new_term_id)
            new_span = KafNafParserPy.Cspan()
            new_span.create_from_ids([token_id])
            term_obj.set_span(new_span)
            naf_obj.add_term(term_obj)
        term_ids.append(term_ids_sub)

    return naf_obj, term_ids


def create_coref_layer(knaf_obj, clusters, term_ids):
    term_ids_list = list(itertools.chain.from_iterable(term_ids))
    for cluster_id, cluster in enumerate(clusters):
        coref_obj = KafNafParserPy.Ccoreference(type=knaf_obj.get_type())
        coref_obj.set_id('co{}'.format(cluster_id + 1))
        coref_obj.set_type('entity')
        for start, end in cluster:
            coref_obj.add_span(term_ids_list[start:end + 1])
            span_text = []
            for term_id in term_ids_list[start:end + 1]:
                word_ids = knaf_obj.get_term(term_id).get_span_ids()
                for word_id in word_ids:
                    word = knaf_obj.get_token(word_id).get_text()
                    span_text.append(word)
            span_text = ' '.join(span_text)
            # TODO: output span_text as comment
        knaf_obj.add_coreference(coref_obj)
    return knaf_obj


def add_linguistic_processors(in_obj):
    name = this_name

    my_lp = KafNafParserPy.Clp()
    my_lp.set_name(name)
    my_lp.set_version(__version__)
    my_lp.set_timestamp()
    in_obj.add_linguistic_processor('coreferences', my_lp)

    return in_obj


def get_jsonlines(knaf_obj):
    sent_term_tok = []

    for term in knaf_obj.get_terms():
        for tok_id in term.get_span_ids():
            tok = knaf_obj.get_token(tok_id)
            sent_term_tok.append(
                (tok.get_sent(), term.get_id(), tok_id, tok.get_text()))

    sentences = []
    term_ids = []
    tok_ids = []
    for sent_id, idlist in itertools.groupby(sent_term_tok, lambda t: t[0]):
        idlist = list(idlist)
        sentences.append([t[3] for t in idlist])
        term_ids.append([t[1] for t in idlist])
        tok_ids.append([t[2] for t in idlist])

    jsonlines_obj = {'doc_key': str(knaf_obj.get_filename()),
                     'sentences': sentences,
                     'clusters': []
                     }
    return jsonlines_obj, term_ids, tok_ids

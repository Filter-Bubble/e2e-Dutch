import tensorflow.compat.v1 as tf
import os
import stanza
import logging

from pathlib import Path

from e2edutch import util
from e2edutch import coref_model as cm
from e2edutch.download import download_data
from e2edutch.predict import Predictor

from stanza.pipeline.processor import Processor, register_processor
from stanza.models.common.doc import Document, Span


# Add a Clusters property to documents as a List of List of Span:
# Clusters is a List of cluster, cluster is a List of Span
def clusterSetter(self, value):
    if isinstance(value, type([])):
        self._clusters = value
    else:
        logger.error('Clusters must be a List')


stanza.models.common.doc.Document.add_property('clusters', default='[]', setter=clusterSetter)


logger = logging.getLogger('e2edutch')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


@register_processor('coref')
class CorefProcessor(Processor):
    ''' Processor that appends coreference information
    Coreferences are added similarly to Stanza's entities:
    * a Document has an attribute clusters that is a List of coreference clusters;
    * a coreference cluster is a List of Stanza Spans.
    '''
    _requires = set(['tokenize'])
    _provides = set(['coref'])

    def __init__(self, config, pipeline, use_gpu):
        # Make e2edutch follow Stanza's GPU settings:
        # set the environment value for GPU, so that initialize_from_env picks it up.
        # if use_gpu:
        #    os.environ['GPU'] = ' '.join(tf.config.experimental.list_physical_devices('GPU'))
        # else:
        #    if 'GPU' in os.environ['GPU'] :
        #        os.environ.pop('GPU')

        self.e2econfig = util.initialize_from_env(model_name='final')

        # Override datapath and log_root:
        # store e2edata with the Stanza resources, ie. a 'stanza_resources/nl/coref' directory
        self.e2econfig['datapath'] = Path(config['model_path']).parent
        self.e2econfig['log_root'] = Path(config['model_path']).parent

        # Download data files if not present
        download_data(self.e2econfig)

        # Start and stop a session to cache all models
        predictor = Predictor(config=self.e2econfig)
        predictor.end_session()

    def _set_up_model(self, *args):
        print('_set_up_model')
        pass

    def process(self, doc):

        predictor = Predictor(config=self.e2econfig)

        # build the example argument for predict:
        #   example (dict): dict with the following fields:
        #                     sentences ([[str]])
        #                     doc_id (str)
        #                     clusters ([[(int, int)]]) (optional)
        example = {}
        example['sentences'] = []
        example['doc_id'] = 'document_from_stanza'  # TODO check what this should be
        example['doc_key'] = 'undocumented'  # TODO check what this should be

        for sentence in doc.sentences:
            s = []
            for word in sentence.words:
                s.append(word.text)
            example['sentences'].append(s)

        predicted_clusters = predictor.predict(example)  # a list of tuples

        # Add the predicted clusters back to the Stanza document

        clusters = []
        for predicted_cluster in predicted_clusters:  # a tuple of entities
            cluster = []
            for predicted_reference in predicted_cluster:  # a tuple of (start, end) word
                start, end = predicted_reference

                # find the sentence_id of the sentence containing this reference
                sentence_id = 0
                sentence = doc.sentences[0]
                sentence_start_word = 0
                sentence_end_word = len(sentence.words) - 1

                while sentence_end_word < start:
                    sentence_start_word = sentence_end_word + 1

                    # move to the next sentence
                    sentence_id += 1
                    sentence = doc.sentences[sentence_id]

                    sentence_end_word = sentence_start_word + len(sentence.words) - 1

                # start counting words from the start of this sentence
                start -= sentence_start_word
                end -= sentence_start_word

                span = Span(  # a list of Tokens
                    tokens=[word.parent for word in sentence.words[start:end + 1]],
                    doc=doc,
                    type='COREF',
                    sent=doc.sentences[sentence_id]
                )
                cluster.append(span)

            clusters.append(cluster)

        doc.clusters = clusters

        predictor.end_session()

        return doc

import os
import stanza
import logging

from pathlib import Path

from e2edutch import util
from e2edutch import coref_model as cm
from e2edutch.download import download_data
from e2edutch.predict import Predictor

from stanza.pipeline.processor import Processor, register_processor
from stanza.models.common.doc import Document

import tensorflow.compat.v1 as tf

logger = logging.getLogger('e2edutch')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


@register_processor('coref')
class CorefProcessor(Processor):
    ''' Processor that appends coreference information '''
    _requires = set(['tokenize'])
    _provides = set(['coref'])
    
    def __init__(self, config, pipeline, use_gpu):
        # Make e2edutch follow Stanza's GPU settings:
        # set the environment value for GPU, so that initialize_from_env picks it up.
        #if use_gpu:
        #    os.environ['GPU'] = ' '.join(tf.config.experimental.list_physical_devices('GPU'))
        #else:
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
        print ('_set_up_model')
        pass

    def process(self, doc):

        predictor = Predictor(config=self.e2econfig)

        # build the example argument for predict:
        #   example (dict): dict with the following fields:
        #                     sentences ([[str]])
        #                     doc_id (str)
        #                     clusters ([[(int, int)]]) (optional)
        example = {}
        example['sentences'] = [sentence.text for sentence in doc.sentences]
        example['doc_id'] = 'document_from_stanza'
        example['doc_key'] = 'undocumented'

        # predicted_clusters, _ = predictor.predict(example)
        print(predictor.predict(example))

        predictor.end_session()

        return doc

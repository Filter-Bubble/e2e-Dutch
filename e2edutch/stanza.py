import stanza
import logging

from e2edutch import util
from e2edutch import coref_model as cm
from e2edutch.download import download_data
from e2edutch.predict import Predictor

from stanza.pipeline.processor import Processor, register_processor
from stanza.models.common.doc import Document

import tensorflow.compat.v1 as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@register_processor('coref')
class CorefProcessor(Processor):
    ''' Processor that appends coreference information '''
    _requires = set(['tokenize'])
    _provides = set(['coref'])
    
    def __init__(self, config, pipeline, use_gpu):
        # Make e2edutch follow Stanza's GPU settings:
        # set the environment value for GPU, so that initialize_from_env picks it up.
        if use_gpu:
            os.environ['GPU'] = ' '.join(tf.config.experimental.list_physical_devices('GPU'))
        else:
            os.environ['GPU'] = ''

        self.e2econfig = util.initialize_from_env(model_name='final')

        # Override datapath:
        # store e2edata with the Stanza resources, ie. a 'stanza_resources/nl/coref' directory
        self.e2econfig['datapath'] = Path(config['model_path']).parent

        # Download data files if not present
        download_data(self.e2econfig)

    def _set_up_model(self, *args):
        print ('_set_up_model')
        pass

    def process(self, doc):
        self.predictor = Predictor(config=self.e2econfig)

        for sent_id, sentence in enumerate(doc.sentences):
            tokenized = ' '.join(w.text for w in sentence.words)

            predicted_clusters, _ = self.predictor.predict(tokenized)
            print(predicted_clusters)

        self.predictor.end_session()

        return doc

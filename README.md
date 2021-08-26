![Python package](https://github.com/Filter-Bubble/e2e-Dutch/workflows/Python%20package/badge.svg)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/Filter-Bubble/e2e-Dutch/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/Filter-Bubble/e2e-Dutch/?branch=master)
[![codecov](https://codecov.io/gh/Filter-Bubble/e2e-coref/branch/master/graph/badge.svg)](https://codecov.io/gh/Filter-Bubble/e2e-coref)
[![DOI](https://zenodo.org/badge/276878416.svg)](https://zenodo.org/badge/latestdoi/276878416)


# e2e-Dutch
Code for e2e coref model in Dutch. The code is based on the [original e2e model for English](https://github.com/kentonl/e2e-coref), and modified to work for Dutch.
If you make use of this code, please [cite it](#citing-this-code) and also cite [the original e2e paper](https://arxiv.org/abs/1804.05392).

This code can be used with a pre-trained model for Dutch, trained on the SoNaR-1 dataset. The model file and documentation can be found at [10.5281/zenodo.5153574](https://zenodo.org/record/5153575)

## Installation
Requirements:
- Python 3.6 or 3.7
- pip
- tensorflow v2.0.0 or higher

In this repository, run:
```
pip install -r requirements.txt
pip install .
```

Alternatively, you can install directly from Pypi:
```
pip install tensorflow
pip install e2e-Dutch
```

## Quick start - Stanza

e2edutch can be used as part of a [Stanza](https://stanfordnlp.github.io/stanza/) pipeline.

Coreferences are added similarly to Stanza's entities:
 * a ___Document___ has an attribute ___clusters___ that is a List of coreference clusters;
 * a coreference cluster is a List of Stanza ___Spans___.

```
import stanza
import e2edutch.stanza

nlp = stanza.Pipeline(lang='nl', processors='tokenize,coref')

doc = nlp('Dit is een test document. Dit document bevat coreferenties.')
print ([[span.text for span in cluster] for cluster in doc.clusters])
```

Note that you first need to download the stanza models with `stanza.download('nl')`.
The e2e-Dutch model files are automatically downloaded to the stanza resources directory when loading the pipeline.

## Quick start
A pretrained model is available to download:
```
python -m e2edutch.download [-d DATAPATH]
```
This downloads the model files, the default location is the `data` directory inside the python package location.
It can also be set manually with the `DATAPATH` argument, or by specifying the enviornment vairable `E2E_HOME`.



The pretrained model can be used to predict coreferences on a conll 2012 files, jsonlines files, [NAF files](https://github.com/newsreader/NAF) or plain text files (in the latter case, the stanza package will be used for tokenization).
```
python -m e2edutch.predict.py [-h] [-o OUTPUT_FILE] [-f {conll,jsonlines,naf}] [-m MODEL] [-c WORD_COL] [--cfg_file CFG_FILE] [--model_cfg_file MODEL_CFG_FILE] [-v] input_filename

positional arguments:
  input_filename

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
  -f {conll,jsonlines,naf}, --format_out {conll,jsonlines,naf}
  -m MODEL, --model MODEL
                        model name
  -c WORD_COL, --word_col WORD_COL
  --cfg_file CFG_FILE   config file
  --model_cfg_file MODEL_CFG_FILE
                        model config file
  -v, --verbose
```
The user-specific configurations (such as data directory, data files, etc) can be provided in a separate config file, the defaults are specified in `cfg/defaults.conf`. The options ` model_cfg_file` and `model` are relevant when you want to use a user-specified model instead of the pretrained model to predict (see the section below on how to train a model).


## Train your own model
To train a new model:
- Make sure the model config file (default: `e2edutch/cfg/models.conf`) describes the model you wish to train
- Make sure your config file (default: `e2edutch/cfg/defaults.conf`) includes the data files you want to use for training
- Run `scripts/setup_train.sh e2edutch/cfg/defaults.conf`. This script converts the conll2012 data to jsonlines files, and caches the word and contextualized embeddings.
- If you want to enable the use of a GPU, set the environment variable:
```bash
export GPU=0
```
- Run the training script:
```bash
python -m e2edutch.train <model-name>
```
## Citing this code
If you use this code in your research, please cite it as follows:
```
@misc{YourReferenceHere,
author = {
            Dafne van Kuppevelt and
            Jisk Attema
         },
title  = {e2e-Dutch},
doi    = {10.5281/zenodo.4146960},
url    = {https://github.com/Filter-Bubble/e2e-Dutch}
}
```
As the code is largely based on [original e2e model for English](https://github.com/kentonl/e2e-coref), please make sure to also cite [the original e2e paper](https://arxiv.org/abs/1804.05392).

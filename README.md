[![Build Status](https://travis-ci.org/Filter-Bubble/e2e-Dutch.svg?branch=master)](https://travis-ci.org/Filter-Bubble/e2e-Dutch)
# e2e-Dutch
Code for e2e coref model in Dutch. The code is based on the [original e2e model for English](https://github.com/kentonl/e2e-coref), and modified to work for Dutch.
If you make use of this code, please also cite [the original e2e paper](https://arxiv.org/abs/1804.05392).

## Installation
Requirements:
- Python 3.6 or 3.7
- pip

In this repository, run:
```
pip install -r requirements.txt
./scripts/setup_all.sh
pip install .
```

The `setup_all` script downloads the word vector files to the `data` directories. 

## Quick start
The e2e-Dutch scripts can take two types of input:
- Files in the [conll-2012 format](http://conll.cemantix.org/2012/data.html)
- `.jsonlines` files, where each line is a json object of a document.

The model configuration are described in the file `cfg/experiments.conf`. To train a new model:
- Make sure the config file describes the model you wish to train
- Run `scripts/setup_train.sh`. This script converts the conll2012 data to jsonlines files, and caches the word and contextualized embeddings.
- Run the training script:
```bash
python scripts/train.py <model-name>
```

A trained model can be used to predict coreferences on a conll 2012 files, json files, or plain text files (in that case, the nltk package will be used for tokenization).
```
python scripts/predict.py [-h] [-o OUTPUT_FILE] [-f {conll,jsonlines}] [-c WORD_COL]
                  config input_filename

positional arguments:
  config
  input_filename

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
  -f {conll,jsonlines}, --format_out {conll,jsonlines}
  -c WORD_COL, --word_col WORD_COL

```

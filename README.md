[![Build Status](https://travis-ci.org/Filter-Bubble/e2e-Dutch.svg?branch=master)](https://travis-ci.org/Filter-Bubble/e2e-Dutch)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/Filter-Bubble/e2e-Dutch/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/Filter-Bubble/e2e-Dutch/?branch=master)
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

The `setup_all` script downloads the word vector files to the `data` directories. It also builds the application-specific tensorflow kernels.

## Quick start
The e2e-Dutch scripts can take two types of input:
- Files in the [conll-2012 format](http://conll.cemantix.org/2012/data.html)
- `.jsonlines` files, where each line is a json object of a document.

The model configuration are described in the file `cfg/models.conf`. The user-specific configurations (such as data directory, data files, etc) can be provided in a separate config file, the defaults are specified in `cfg/defaults.conf`.


To train a new model:
- Make sure the model config file describes the model you wish to train
- Make sure your config file includes the data files you want to use for training
- Run `scripts/setup_train.sh e2edutch/cfg/defaults.conf`. This script converts the conll2012 data to jsonlines files, and caches the word and contextualized embeddings.
- If you want to enable the use of a GPU, set the environment variable:
```bash
export GPU=0
```
- Run the training script:
```bash
python scripts/train.py <model-name>
```

A trained model can be used to predict coreferences on a conll 2012 files, json files, [NAF files](https://github.com/newsreader/NAF) or plain text files (in the latter case, the nltk package will be used for tokenization).
```
predict.py [-h] [-o OUTPUT_FILE] [-f {conll,jsonlines,naf}]
                  [-c WORD_COL] [--cfg_file CFG_FILE] [-v]
                  config input_filename

positional arguments:
  config: name of the model to use for prediction
  input_filename

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
  -f {conll,jsonlines,naf}, --format_out {conll,jsonlines,naf}
  -c WORD_COL, --word_col WORD_COL
  --cfg_file CFG_FILE   config file
  -v, --verbose


```

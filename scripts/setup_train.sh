#!/bin/bash

dlx() {
  wget $1/$2
  tar -xvzf $2
  rm $2
}

dlx http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz
mkdir conll-2012
mv reference-coreference-scorers conll-2012/scorer

# TODO: download conll data
python e2edutch/minimize.py data/train.dutch.conll
python e2edutch/minimize.py data/dev.dutch.conll

python scripts/char_vocab.py data/train.dutch.jsonlines data/dev.dutch.jsonlines data/char_vocab.dutch.txt

# Filter word embeddings
python scripts/filter_embeddings.py -c data/fasttext.300.vec data/train.dutch.jsonlines data/dev.dutch.jsonlines data/fasttext.300.vec.filtered

# Cache BERT embeddings
python scripts/cache_bert.py bertje data/train.dutch.jsonlines data/dev.dutch.jsonlines

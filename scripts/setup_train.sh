#!/bin/bash
CONFIG_FILE=$1


extract_config() {
  pyhocon -f properties < $CONFIG_FILE | awk '/^'$1'/{print $3}'
}

dlx() {
  wget $1/$2
  tar -xvzf $2
  rm $2
}

TRAIN_FILE_JSON=`extract_config train_path`
TRAIN_FILE_CONLL=`extract_config conll_train_path`
DEV_FILE_JSON=`extract_config eval_path`
DEV_FILE_CONLL=`extract_config conll_eval_path`
DATAPATH=`extract_config datapath`

echo $TRAIN_FILE_JSON $TRAIN_FILE_CONLL $DEV_FILE_JSON $DEV_FILE_CONLL
dlx http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz
mkdir conll-2012
mv reference-coreference-scorers conll-2012/scorer

# TODO: download conll data
python e2edutch/minimize.py $TRAIN_FILE_CONLL -o $TRAIN_FILE_JSON
python e2edutch/minimize.py $DEV_FILE_CONLL -o $DEV_FILE_JSON

python scripts/char_vocab.py $TRAIN_FILE_JSON $DEV_FILE_JSON $DATAPATH/char_vocab.dutch.txt

# Filter word embeddings
python scripts/filter_embeddings.py -c $DATAPATH/fasttext.300.vec $TRAIN_FILE_JSON $DEV_FILE_JSON $DATAPATH/fasttext.300.vec.filtered

# Cache BERT embeddings
python scripts/cache_bert.py bertje $DATAPATH $TRAIN_FILE_JSON $DEV_FILE_JSON

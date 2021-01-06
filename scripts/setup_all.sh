# Make data directory
mkdir -p data

if [ ! -f data/fasttext.300.vec ]; then
  wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz
  gzip -d cc.nl.300.vec.gz
  tail -n +2 cc.nl.300.vec > data/fasttext.300.vec
  rm cc.nl.300.vec
fi


# Download BERT-NL model
# uncomment if using BERT NL model (default is Bertje)
# if [ ! -f data/bert-nl ]; then
#   wget http://textdata.nl/bert-nl/dutch_cased_punct_L-12_H-768_A-12.zip
#   unzip dutch_cased_punct_L-12_H-768_A-12.zip -x "dutch_cased_punct_L-12_H-768_A-12/*"
#   mv dutch_cased_punct_L-12_H-768_A-12-NEW data/bert-nl
#   rm dutch_cased_punct_L-12_H-768_A-12.zip
# fi

# Download trained e2e model_
if [ ! -f data/best/model.max.ckpt.index ]; then
  wget https://surfdrive.surf.nl/files/index.php/s/UnZMyDrBEFunmQZ/download -O model.zip
  unzip model.zip
fi

# Make data directory
mkdir -p data

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz
gzip -d cc.nl.300.vec.gz
tail -n +2 cc.nl.300.vec > data/fasttext.300.vec
rm cc.nl.300.vec


# Download BERT-NL model
# wget http://textdata.nl/bert-nl/dutch_cased_punct_L-12_H-768_A-12.zip
# unzip dutch_cased_punct_L-12_H-768_A-12.zip -x "dutch_cased_punct_L-12_H-768_A-12/*"
# mv dutch_cased_punct_L-12_H-768_A-12-NEW data/bert-nl
# rm dutch_cased_punct_L-12_H-768_A-12.zip

# Build custom kernels.
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Linux
mkdir -p e2edutch/lib
g++ -std=c++11 -shared e2edutch/coref_kernels.cc -o e2edutch/lib/coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

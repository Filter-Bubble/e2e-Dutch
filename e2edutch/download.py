import requests
import gzip
import zipfile
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
from e2edutch import util

logger = logging.getLogger()


def download_file(url, path):
    """
    Download a URL into a file as specified by `path`.
    """
    # This function is copied from stanza
    # https://github.com/stanfordnlp/stanza/blob/f0338f891a03e242c7e11e440dec6e191d54ab77/stanza/resources/common.py#L103
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        file_size = int(r.headers.get('content-length'))
        default_chunk_size = 131072
        desc = 'Downloading ' + url
        with tqdm(total=file_size, unit='B', unit_scale=True,
                  desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=default_chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    pbar.update(len(chunk))


def download_data(config={}):
    # Create the data directory if it doesn't exist yet
    data_dir = Path(config['datapath'])
    logger.info('Downloading to {}'.format(data_dir))
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download word vectors
    logger.info('Download word vectors')
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz"
    fname = data_dir / 'fasttext.300.vec'
    fname_gz = data_dir / 'fasttext.300.vec.gz'
    if not fname.exists():
        download_file(url, fname_gz)
        with gzip.open(fname_gz, 'rb') as fin:
            with open(fname, 'wb') as fout:
                # We need to remove the first line
                for i, line in enumerate(fin.readlines()):
                    if i > 0:
                        fout.write(line)
        # Remove gz file
        fname_gz.unlink()
    else:
        logger.info('Word vectors file already exists')

    # Download e2e dutch model_
    logger.info('Download e2e model')
    url = "https://surfdrive.surf.nl/files/index.php/s/UnZMyDrBEFunmQZ/download"
    fname_zip = data_dir / 'model.zip'
    log_dir_name = data_dir / 'final'
    model_file = log_dir_name / 'model.max.ckpt.index'
    if not fname_zip.exists() and not model_file.exists():
        download_file(url, fname_zip)
    if not model_file.exists():
        with zipfile.ZipFile(fname_zip, 'r') as zfile:
            zfile.extractall(data_dir)
        Path(data_dir / 'logs' / 'final').rename(log_dir_name)
        Path(data_dir, 'logs').rmdir()
    else:
        logger.info('E2e model file already exists')

    # Download char_dict
    logger.info('Download char dict')
    url = "https://github.com/Filter-Bubble/e2e-Dutch/raw/v0.2.0/data/char_vocab.dutch.txt"
    fname = data_dir / 'char_vocab.dutch.txt'
    if not fname.exists():
        download_file(url, fname)
    else:
        logger.info('Char dict file already exists')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datapath', default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.verbose:
        # logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)
    # To do: argparse for config file
    if args.datapath is None:
        config = util.initialize_from_env(model_name='final')
    else:
        config = {'datapath': args.datapath}
    download_data(config)


if __name__ == "__main__":
    main()

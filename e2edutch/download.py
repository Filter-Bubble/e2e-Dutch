import os
import requests
import gzip
import zipfile
from tqdm import tqdm
from pathlib import Path
from e2edutch import util


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
    data_dir = Path(util.get_data_dir(config))
    # Create the data directory if it doesn't exist yet
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download word vectors
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

    # Download e2e dutch model_
    url = "https://surfdrive.surf.nl/files/index.php/s/UnZMyDrBEFunmQZ/download"
    fname_zip = data_dir / 'model.zip'
    log_dir_name = data_dir / 'final'
    if not fname_zip.exists() and not log_dir_name.exists():
        download_file(url, fname_zip)
    if not log_dir_name.exists():
        with zipfile.ZipFile(fname_zip, 'r') as zfile:
            zfile.extractall(data_dir)
        Path(data_dir / 'logs' / 'final').rename(log_dir_name)
        Path(data_dir, 'logs').rmdir()

    # Download char_dict
    url = "https://github.com/Filter-Bubble/e2e-Dutch/raw/v0.2.0/data/char_vocab.dutch.txt"
    fname = data_dir / 'char_vocab.dutch.txt'
    if not fname.exists():
        download_file(url, fname)


def main():
    # To do: argparse for config file
    download_data()


if __name__ == "__main__":
    main()

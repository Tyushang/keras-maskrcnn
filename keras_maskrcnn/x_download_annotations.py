#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"
import argparse
import os
import urllib.parse
import urllib.request
import tarfile


CONFIG = {
    'download_dir': './oid/',
    'tarfile_path': None,
}

DOWNLOAD_DESC = {
    # images for train/validation/test
    'images': None,
    # Object Detection track anno
    'annotation-object-detection': None,
    # Instance Segmentation track anno
    'annotation-instance-segmentation': {
        'train': {
            'challenge-2019-train-segmentation-labels.csv': 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-segmentation-labels.csv',
            'challenge-2019-train-segmentation-bbox.csv': 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-segmentation-bbox.csv',
            'challenge-2019-train-masks': [
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-0.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-1.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-2.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-3.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-4.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-5.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-6.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-7.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-8.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-9.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-a.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-b.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-c.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-d.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-e.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-f.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-segmentation-masks.csv',
            ]
        },
        'validation': {
            'challenge-2019-validation-segmentation-labels.csv': 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-validation-segmentation-labels.csv',
            'challenge-2019-validation-segmentation-bbox.csv': 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-validation-segmentation-bbox.csv',
            'challenge-2019-validation-masks': [
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-0.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-1.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-2.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-3.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-4.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-5.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-6.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-7.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-8.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-9.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-a.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-b.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-c.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-d.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-e.zip',
                # 'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-f.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-validation-segmentation-masks.csv',
            ],
        },
        'metadata': {
            'challenge-2019-classes-description-segmentable.csv': 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-classes-description-segmentable.csv',
            'challenge-2019-label300-segmentable-hierarchy.json': 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-label300-segmentable-hierarchy.json',
        },
    },
    # Visual Relationships Detection track anno
    'annotation-visual-relationships-detection': None,
}

ABS_DOWNLOAD_DIR = os.path.abspath(DOWNLOAD_DESC['download_dir'])
ABS_TAR_PATH = os.path.abspath(DOWNLOAD_DESC['tarfile_path'])


def is_url(s: str):
    return s.startswith('https://')


def url_filename(s: str):
    res = urllib.parse.urlparse(s)
    return res.path.split('/')[-1]


def download(desc):

    def _download(k, v):
        if v is None:
            return
        elif type(v) == dict:
            os.makedirs(k, exist_ok=True)
            os.chdir(k)
            for k2, v2 in v.items():
                _download(k2, v2)
            os.chdir('..')
        elif type(v) == list:
            os.makedirs(k, exist_ok=True)
            os.chdir(k)
            for e in v:
                urllib.request.urlretrieve(e, url_filename(e))
            os.chdir('..')
        elif is_url(v):
            urllib.request.urlretrieve(v, k)

    pwd = os.getcwd()
    try:
        if not os.path.exists(CONFIG['download_dir']):
            os.makedirs(CONFIG['download_dir'])
        os.chdir(CONFIG['download_dir'])
        for k, v in desc.items():
            _download(k, v)
    finally:
        os.chdir(pwd)


def tar(dir_to_tar, abs_tar_path):
    basename = os.path.basename(abs_tar_path)
    os.chdir(os.path.dirname(abs_tar_path))
    with tarfile.open(basename, 'w') as tar:
        tar.add(dir_to_tar, arcname=basename.split('.')[0])


if __name__ == '__main__':
    # use hard-coded CONFIG if it defined, else, use CLI.
    if 'CONFIG' not in dir():
        parser = argparse.ArgumentParser(description='Set download dir, and tarfile path(optional).'
                                                     'files to be downloaded, url and path-tree '
                                                     'are hard-coded by DOWNLOAD_DESC.')
        parser.add_argument('--download-dir', required=True, default='./oid', help='Set download dir.')
        parser.add_argument('--tarfile-path', help='Set tarfile(optional), if not set, do not tar.')
        args = parser.parse_args()
        CONFIG = args.__dict__

    download(DOWNLOAD_DESC)
    if 'tarfile_path' in CONFIG.keys():
        tar(ABS_DOWNLOAD_DIR, ABS_TAR_PATH)



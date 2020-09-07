#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"


import os

import cv2
from albumentations import *

from utils.ins_utils import CSVGenerator

os.chdir(r'D:\venv-tensorflow2\keras-maskrcnn')
DATASET_DIR = '../open-images-dataset'
ANNO_PATH   = DATASET_DIR + '/annotation-instance-segmentation/'
BATCH_SIZE  = 4

bbox_params = BboxParams(format='pascal_voc', min_area=1.0, min_visibility=0.1, label_fields=['labels'])
transform_generator = Compose([
    HorizontalFlip(p=0.9),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.1, border_mode=cv2.BORDER_CONSTANT),
], bbox_params=bbox_params, p=1.0)


tg = CSVGenerator(
    ANNO_PATH + f'train/challenge-2019-train-masks/challenge-2019-train-segmentation-masks.csv',
    ANNO_PATH + 'metadata/challenge-2019-classes-description-segmentable.csv',
    image_dir=f'{DATASET_DIR}/train',
    transform_generator=transform_generator,
    batch_size=BATCH_SIZE,
)
for i in range(10):
    tt = tg[i]

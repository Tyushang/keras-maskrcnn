#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import cv2
from albumentations import *

from utils.ins_utils import CSVGenerator
from utils.visualization import draw_mask, draw_masks

DATASET_DIR     = 'D:/venv-tensorflow2/open-images-dataset'
# DATASET_DIR     = 'gs://tyu-ins-sample'
MASK_CSV_PATH   = f'{DATASET_DIR}/annotation-instance-segmentation/train/' \
                  f'challenge-2019-train-masks/challenge-2019-train-segmentation-masks.csv'
CLASS_CSV_PATH  = f'{DATASET_DIR}/annotation-instance-segmentation/' \
                  f'metadata/challenge-2019-classes-description-segmentable.csv'

BATCH_SIZE = 4

import pandas as pd

mask_df = pd.read_csv(MASK_CSV_PATH)

bbox_params = BboxParams(format='pascal_voc', min_area=1.0, min_visibility=0.1, label_fields=['labels'])
transform_generator = Compose([
    HorizontalFlip(p=1),
    # OneOf([
    #     IAAAdditiveGaussianNoise(),
    #     GaussNoise(),
    # ], p=0.1),
    # OneOf([
    #     MotionBlur(p=.1),
    #     MedianBlur(blur_limit=3, p=.1),
    #     Blur(blur_limit=3, p=.1),
    # ], p=0.1),
    # ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.1, border_mode=cv2.BORDER_CONSTANT),
    # OneOf([
    #     CLAHE(clip_limit=2),
    #     IAASharpen(),
    #     IAAEmboss(),
    #     RandomBrightnessContrast(),
    # ], p=0.1),
    # OneOf([
    #     RGBShift(p=1.0, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
    #     HueSaturationValue(p=1.0),
    # ], p=0.1),
    # ToGray(p=0.01),
    # ImageCompression(p=0.05, quality_lower=50, quality_upper=99),
], bbox_params=bbox_params, p=1.0)

gen = CSVGenerator(
    MASK_CSV_PATH,
    CLASS_CSV_PATH,
    image_dir=os.path.join(DATASET_DIR, 'train'),
    transform_generator=transform_generator,
    batch_size=BATCH_SIZE,
    config=None,
    image_min_side=800,
    image_max_side=1024,
    group_method='random',
    is_rle=False
)


def show(image_batch, anno_batch):
    for image, boxes, labels, masks in zip(image_batch, *anno_batch):
        draw_masks(image, boxes, masks )
        print(image.shape)
        print(boxes.shape)
        print(labels.shape)
        print(masks.shape)
        break



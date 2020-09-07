#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script sample some examples(image and annotations) from origin dataset for debug use.
Because origin dataset is too large.

Dataset Dir Tree:
ROOT:
|-- annotation-instance-segmentation
|   |-- metadata
|   |   |-- challenge-2019-classes-description-segmentable.csv  # class_csv for short.
|   |   |-- challenge-2019-label300-segmentable-hierarchy.json  # hierarchy_json for short
|   |-- train
|   |   |-- challenge-2019-train-masks
|   |   |   |-- challenge-2019-train-segmentation-masks.csv     # mask_csv for short.
|   |   |   |-- challenge-2019-train-masks-[0~f].zip
|   |   |-- all-masks                                           # N_MASK: 2125530
|   |   |-- challenge-2019-train-segmentation-bbox.csv          # bbox_csv for short.
|   |   |-- challenge-2019-train-segmentation-labels.csv        # label_csv for short.
|   |-- validation
|       |-- challenge-2019-validation-masks
|       |   |-- challenge-2019-validation-segmentation-masks.csv
|       |   |-- challenge-2019-validation-masks-[0~f].zip
|       |-- all-masks                                           # N_MASK: 23366
|       |-- challenge-2019-validation-segmentation-bbox.csv
|       |-- challenge-2019-validation-segmentation-labels.csv
|-- train       # N_IMAGE
|-- validation  # N_IMAGE
|-- test        # N_IMAGE
"""


import os
import random
import pandas as pd


SRC_ROOT = 'gs://tyu-ins'
DST_ROOT = '/tmp/tyu-ins-sample'

N_SAMPLE_TRAIN      = 300
N_SAMPLE_VALIDATION = 100
N_SAMPLE_TEST       = 100


def row_filter(df: pd.DataFrame, on: str, to_stay: list):
    other_df = pd.DataFrame({on: to_stay})
    sampled_df = df.set_index(on).join(other_df.set_index(on), how='inner')
    return sampled_df.reset_index()


if SRC_ROOT and DST_ROOT:
    if 'annotation-instance-segmentation':
        if 'metadata':
            src = f'{SRC_ROOT}/annotation-instance-segmentation/metadata'
            dst = f'{DST_ROOT}/annotation-instance-segmentation/'
            os.makedirs(dst, exist_ok=True)
            !gsutil cp -r $src $dst
        if 'train':
            if 'challenge-2019-train-masks':
                if 'challenge-2019-train-segmentation-masks.csv':
                    src = f'{SRC_ROOT}/annotation-instance-segmentation/train/challenge-2019-train-masks/challenge-2019-train-segmentation-masks.csv'
                    dst = f'{DST_ROOT}/annotation-instance-segmentation/train/challenge-2019-train-masks/challenge-2019-train-segmentation-masks.csv'
                    # columns: MaskPath	ImageID	LabelName	BoxID	BoxXMin	BoxXMax	BoxYMin	BoxYMax	PredictedIoU	Clicks
                    mask_df_train      = pd.read_csv(src)
                    dedup_ids_train    = list(set(mask_df_train['ImageID']))
                    SAMPLED_IDS_TRAIN  = random.choices(dedup_ids_train, k=N_SAMPLE_TRAIN)
                    sampled_df_train   = row_filter(mask_df_train, on='ImageID', to_stay=SAMPLED_IDS_TRAIN)
                    SAMPLED_MASK_TRAIN = sampled_df_train['MaskPath']  # MaskPath actually is Mask fname.
                    sampled_df_train.to_csv('./challenge-2019-train-segmentation-masks.csv', index=False)
                    !gsutil cp './challenge-2019-train-segmentation-masks.csv' $dst
            if 'all-masks':
                src = list(map(lambda fname: f'{SRC_ROOT}/annotation-instance-segmentation/train/all-masks/{fname}', SAMPLED_MASK_TRAIN))
                dst = f'{DST_ROOT}/annotation-instance-segmentation/train/all-masks/'
                with open('./sampled_mask_paths_train.txt', 'w') as f:
                    f.write('\n'.join(src))
                os.makedirs(dst, exist_ok=True)
                !cat './sampled_mask_paths_train.txt' | gsutil -q -m cp -I $dst
            if 'challenge-2019-train-segmentation-bbox.csv':
                src = f'{SRC_ROOT}/annotation-instance-segmentation/train/challenge-2019-train-segmentation-bbox.csv'
                dst = f'{DST_ROOT}/annotation-instance-segmentation/train/challenge-2019-train-segmentation-bbox.csv'
                # columns: ImageID	LabelName	XMin	XMax	YMin	YMax	IsGroupOf
                bbox_df_train    = pd.read_csv(src)
                sampled_df_train = row_filter(bbox_df_train, on='ImageID', to_stay=SAMPLED_IDS_TRAIN)
                sampled_df_train.to_csv('./challenge-2019-train-segmentation-bbox.csv', index=False)
                !gsutil cp './challenge-2019-train-segmentation-bbox.csv' $dst
            if 'challenge-2019-train-segmentation-labels.csv':
                src = f'{SRC_ROOT}/annotation-instance-segmentation/train/challenge-2019-train-segmentation-labels.csv'
                dst = f'{DST_ROOT}/annotation-instance-segmentation/train/challenge-2019-train-segmentation-labels.csv'
                # columns: ImageID	LabelName	Confidence
                label_df_train   = pd.read_csv(src)
                sampled_df_train = row_filter(label_df_train, on='ImageID', to_stay=SAMPLED_IDS_TRAIN)
                sampled_df_train.to_csv('./challenge-2019-train-segmentation-labels.csv', index=False)
                !gsutil cp './challenge-2019-train-segmentation-labels.csv' $dst
        if 'validation':
            if 'challenge-2019-validation-masks':
                if 'challenge-2019-validation-segmentation-masks.csv':
                    src = f'{SRC_ROOT}/annotation-instance-segmentation/validation/challenge-2019-validation-masks/challenge-2019-validation-segmentation-masks.csv'
                    dst = f'{DST_ROOT}/annotation-instance-segmentation/validation/challenge-2019-validation-masks/challenge-2019-validation-segmentation-masks.csv'
                    # columns: MaskPath	ImageID	LabelName	BoxID	BoxXMin	BoxXMax	BoxYMin	BoxYMax	PredictedIoU	Clicks
                    mask_df_valid      = pd.read_csv(src)
                    dedup_ids_valid    = list(set(mask_df_valid['ImageID']))
                    SAMPLED_IDS_VALID  = random.choices(dedup_ids_valid, k=N_SAMPLE_VALIDATION)
                    sampled_df_valid   = row_filter(mask_df_valid, on='ImageID', to_stay=SAMPLED_IDS_VALID)
                    SAMPLED_MASK_VALID = sampled_df_valid['MaskPath']
                    sampled_df_valid.to_csv('./challenge-2019-validation-segmentation-masks.csv', index=False)
                    !gsutil cp './challenge-2019-validation-segmentation-masks.csv' $dst
            if 'all-masks':
                src = list(map(lambda fname: f'{SRC_ROOT}/annotation-instance-segmentation/validation/all-masks/{fname}', SAMPLED_MASK_VALID))
                dst = f'{DST_ROOT}/annotation-instance-segmentation/validation/all-masks/'
                with open('./sampled_mask_paths_valid.txt', 'w') as f:
                    f.write('\n'.join(src))
                os.makedirs(dst, exist_ok=True)
                !cat './sampled_mask_paths_valid.txt' | gsutil -q -m cp -I $dst
            if 'challenge-2019-validation-segmentation-bbox.csv':
                src = f'{SRC_ROOT}/annotation-instance-segmentation/validation/challenge-2019-validation-segmentation-bbox.csv'
                dst = f'{DST_ROOT}/annotation-instance-segmentation/validation/challenge-2019-validation-segmentation-bbox.csv'
                # columns: ImageID	LabelName	XMin	XMax	YMin	YMax	IsGroupOf
                bbox_df_valid    = pd.read_csv(src)
                sampled_df_valid = row_filter(bbox_df_valid, on='ImageID', to_stay=SAMPLED_IDS_VALID)
                sampled_df_valid.to_csv('./challenge-2019-validation-segmentation-bbox.csv', index=False)
                !gsutil cp './challenge-2019-validation-segmentation-bbox.csv' $dst
            if 'challenge-2019-validation-segmentation-labels.csv':
                src = f'{SRC_ROOT}/annotation-instance-segmentation/validation/challenge-2019-validation-segmentation-labels.csv'
                dst = f'{DST_ROOT}/annotation-instance-segmentation/validation/challenge-2019-validation-segmentation-labels.csv'
                # columns: ImageID	LabelName	Confidence
                label_df_valid   = pd.read_csv(src)
                sampled_df_valid = row_filter(label_df_valid, on='ImageID', to_stay=SAMPLED_IDS_VALID)
                sampled_df_valid.to_csv('./challenge-2019-validation-segmentation-labels.csv', index=False)
                !gsutil cp './challenge-2019-validation-segmentation-labels.csv' $dst
    if 'train':
        src = list(map(lambda id: f'{SRC_ROOT}/train/{id}.jpg', SAMPLED_IDS_TRAIN))
        dst = f'{DST_ROOT}/train/'
        with open('./sampled_id_paths_train.txt', 'w') as f:
            f.write('\n'.join(src))
        os.makedirs(dst, exist_ok=True)
        !cat './sampled_id_paths_train.txt' | gsutil -q -m cp -I $dst
    if 'validation':
        src = list(map(lambda id: f'{SRC_ROOT}/validation/{id}.jpg', SAMPLED_IDS_VALID))
        dst = f'{DST_ROOT}/validation/'
        with open('./sampled_id_paths_valid.txt', 'w') as f:
            f.write('\n'.join(src))
        os.makedirs(dst, exist_ok=True)
        !cat './sampled_id_paths_valid.txt' | gsutil -q -m cp -I $dst
    if 'test':
        src = os.popen(f'gsutil ls {SRC_ROOT}/test').read().split('\n')[:-1]
        dst = f'{DST_ROOT}/test/'
        sampled_id_paths_test = random.choices(src, k=N_SAMPLE_TEST)
        with open('sampled_id_paths_test.txt', 'w') as f:
            f.write('\n'.join(sampled_id_paths_test))
        os.makedirs(dst, exist_ok=True)
        !cat './sampled_id_paths_test.txt' | gsutil -q -m cp -I $dst




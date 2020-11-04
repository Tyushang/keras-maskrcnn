#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

# Dataset Dir Tree:
# ROOT:
# |-- annotation-instance-segmentation
# |   |-- metadata
# |   |   |-- challenge-2019-classes-description-segmentable.csv  # class_csv for short.
# |   |   |-- challenge-2019-label300-segmentable-hierarchy.json  # hierarchy_json for short
# |   |-- train
# |   |   |-- challenge-2019-train-masks
# |   |   |   |-- challenge-2019-train-segmentation-masks.csv     # mask_csv for short.
# |   |   |   |-- challenge-2019-train-masks-[0~f].zip
# |   |   |-- all-masks                                           # N_MASK: 2125530
# |   |   |-- challenge-2019-train-segmentation-bbox.csv          # bbox_csv for short.
# |   |   |-- challenge-2019-train-segmentation-labels.csv        # label_csv for short.
# |   |-- validation
# |       |-- challenge-2019-validation-masks
# |       |   |-- challenge-2019-validation-segmentation-masks.csv
# |       |   |-- challenge-2019-validation-masks-[0~f].zip
# |       |-- all-masks                                           # N_MASK: 23366
# |       |-- challenge-2019-validation-segmentation-bbox.csv
# |       |-- challenge-2019-validation-segmentation-labels.csv
# |-- train       # N_IMAGE
# |-- validation  # N_IMAGE
# |-- test        # N_IMAGE

import os
import pandas as pd

# tv: train or validation
tv = 'train'

# Get Sample ids
urls_path = f'../open-images-dataset/{tv}-samples/urls.txt'
ids = list(map(lambda s: os.path.basename(s).split('.')[0],  open(urls_path, 'r').readlines()))

# Select mask sample anno by ids
mask_path = f'../open-images-dataset/annotation-instance-segmentation/{tv}/challenge-2019-{tv}-masks/challenge-2019-{tv}-segmentation-masks.csv'
mask_df   = pd.read_csv(mask_path)
select = pd.Series([True if id in ids else False for id in mask_df['ImageID']])
mask_sample_df = mask_df[select]

# Save mask sample anno
mask_sample_path = os.path.dirname(mask_path) + f'/challenge-2019-{tv}-segmentation-masks-samples.csv'
mask_sample_df.to_csv(mask_sample_path, index=False)

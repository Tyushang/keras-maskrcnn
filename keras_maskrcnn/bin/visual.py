#!/usr/bin/env python3
# -*- coding: utf-8 -*-

DATASET_DIR = 'D:/venv-tensorflow2/open-images-dataset'
MASK_CSV_PATH = f'{DATASET_DIR}/annotation-instance-segmentation/train/' \
                f'challenge-2019-train-masks/challenge-2019-train-segmentation-masks.csv'

import pandas as pd

mask_df = pd.read_csv(MASK_CSV_PATH)






# coding: utf-8


__author__ = 'Frank Jing'

# _________________________________________________________________________________________________
# Configurations:
import os
import random
import glob

RUN_ON = 'kaggle' if os.path.exists('/kaggle') else 'local'

if RUN_ON == 'kaggle':
    MODEL_PATH = '../input/download-models/resnet50_oid_v1.0.1.h5'
    TEST_IMAGE_PATHS = random.choices(glob.glob('../input/open-images-2019-instance-segmentation/test/*.jpg'), k=4)
else:
    os.chdir('D:/venv-tensorflow2/keras-maskrcnn')
    MODEL_PATH = '../ins/model.h5'
    TEST_IMAGE_PATHS = glob.glob('../keras-mrcnn-tf2/sample_images/*.jpg')

BATCH_SIZE = 4
BACKBONE   = 'resnet50'

# _________________________________________________________________________________________________
# #### Configurations that do not need to change:

# _________________________________________________________________________________________________
# Program:

# noinspection PyUnresolvedReferences
import tensorflow as tf

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_maskrcnn import models
from utils.ins_utils import *

# Set your own project id here
PROJECT_ID = 'banded-splicer-259715'
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)


def read_jpg(path):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw)
    res = tf.image.resize(img, size=[1024, 1024])
    return res


if __name__ == '__main__':
    model: tf.keras.Model = models.load_model(MODEL_PATH, backbone_name=BACKBONE)

    ds0 = tf.data.Dataset.from_tensor_slices(TEST_IMAGE_PATHS)

    ds1 = ds0.map(read_jpg).batch(BATCH_SIZE)

    x = next(iter(ds1))

    pred = model.predict_on_batch(x)

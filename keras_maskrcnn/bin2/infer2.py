# coding: utf-8


__author__ = 'Frank Jing'

# _________________________________________________________________________________________________
# Configurations:
import os
import random
import glob

# noinspection PyUnresolvedReferences
import tensorflow as tf

import keras_retinanet.losses
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.utils.tf_version import check_tf_version
from keras_retinanet.utils.model import freeze as freeze_model

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_maskrcnn import losses
from keras_maskrcnn import models
from albumentations import *
#
from keras_maskrcnn.utils.ins_utils import *


RUN_ON = 'kaggle' if os.path.exists('/kaggle') else 'local'

if RUN_ON == 'kaggle':
    MODEL_PATH = '../input/download-models/resnet50_oid_v1.0.1.h5'
    TEST_IMAGE_PATHS = glob.glob('../input/open-images-2019-instance-segmentation/test/*.jpg')
else:
    os.chdir(r'D:\venv-tensorflow2\keras-maskrcnn')
    MODEL_PATH = '../ins/model.h5'
    TEST_IMAGE_PATHS = glob.glob('../keras-mrcnn-tf2/sample_images/*.jpg')

BATCH_SIZE = 2
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
    return tf.image.decode_jpeg(tf.io.read_file(path))

def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model

def create_models(backbone_retinanet, num_classes, weights, freeze_backbone=False, class_specific_filter=True, anchor_params=None):
    from keras_retinanet.utils.model import freeze as freeze_model
    modifier = freeze_model if freeze_backbone else None

    model = model_with_weights(
        backbone_retinanet(
            num_classes,
            nms=True,
            class_specific_filter=class_specific_filter,
            modifier=modifier,
            anchor_params=anchor_params
        ), weights=weights, skip_mismatch=True)
    training_model   = model
    prediction_model = model

    # compile model
    opt = tf.keras.optimizers.Adam(lr=1e-5, )  # clipnorm=0.001)

    # compile model
    training_model.compile(
        loss={
            'regression_cat'    : keras_retinanet.losses.smooth_l1(),
            'classification_cat': keras_retinanet.losses.focal(),
            'masks'         : losses.mask(),
        },
        optimizer=opt,
    )
    model.run_eagerly = True

    return model, training_model, prediction_model



if __name__ == '__main__':
    if 'model' not in dir():
        # model: tf.keras.Model = models.load_model(MODEL_PATH, backbone_name=BACKBONE)
        backbone = models.backbone('resnet50')
        model, *_ = create_models(
            backbone_retinanet=backbone.maskrcnn,
            num_classes=300,
            weights='../ins/weights/mask_rcnn_resnet50_oid_v1.0.h5',
            class_specific_filter=False,
            anchor_params=None
        )
        model.run_eagerly = True
    ds0 = tf.data.Dataset.from_tensor_slices(TEST_IMAGE_PATHS)
    ds1 = ds0.map(read_jpg).padded_batch(BATCH_SIZE)
    inp = list(ds1.take(1))[0]
    # pred = model.predict_on_batch(inp)
    import tensorflow.keras.backend as K

    def gl(*layer_names):
        ret = model
        for name in layer_names:
            ret = ret.get_layer(name)
        return ret

    k_func = K.function(inputs=[model.layers[0].input, ],
                        # outputs=[GL(name).output for name in ['P3', 'P4', 'P5', 'P6', 'P7']]
                        # outputs=[GL(name).input for name in ['C3_reduced', 'C4_reduced', 'C5_reduced', ]],
                        outputs=[model.output, ],
                        )
    res = k_func(inp)

    for r in res:
        if isinstance(res, (list, tuple)):
            for r2 in r:
                print(r2.shape)
        else:
            print(r.shape)

# coding: utf-8

# _________________________________________________________________________________________________
# argparse:
import os, sys, argparse


parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet mask network.')
subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
subparsers.required = True

coco_parser = subparsers.add_parser('coco')
coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

csv_parser = subparsers.add_parser('csv')
csv_parser.add_argument('dataset_dir', help='Path to OID training images.')
csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
csv_parser.add_argument('class_names', help='Path to a CSV file containing class label mapping.')
csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

group = parser.add_mutually_exclusive_group()
group.add_argument('--snapshot',          help='Resume training from a snapshot.')
group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
group.add_argument('--weights',           help='Initialize the model with weights from a file.')
group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
parser.add_argument('--fold',             help='Fold number.', type=int, default=1)
parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
parser.add_argument('--accum_iters',      help='Accum iters. If more than 1 used AdamAccum optimizer', type=int, default=1)
parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.', dest='class_specific_filter', action='store_false')
parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among mid_to_no.', action='store_true')
parser.add_argument('--group_method',     help='How to form batches', default='random')

# _________________________________________________________________________________________________
# Configurations:
import tensorflow as tf

RUN_ON = 'local' if os.path.exists('C:/') else \
         'kaggle' if os.path.exists('/kaggle') else \
         'gcp'

GPU = tf.config.experimental.list_physical_devices('GPU')

try:  # Detect hardware, return appropriate distribution strategy
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', TPU.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    # instantiate a distribution strategy
    STRATEGY = tf.distribute.experimental.TPUStrategy(TPU)
except ValueError:
    TPU      = None
    # STRATEGY = tf.distribute.MirroredStrategy() if GPU else\
    #            tf.distribute.get_strategy()
    STRATEGY = tf.distribute.get_strategy()

if '--csv' in sys.argv:
    # if cli is used, set config by cli args.
    CONFIG = parser.parse_args()
else:
    if RUN_ON == 'local':
        os.chdir(r'D:\venv-tensorflow2\keras-maskrcnn')
        dir_dataset        = '../open-images-dataset'
        # https://github.com/fizyr/keras-maskrcnn/releases/download/0.2.2/resnet50_oid_v1.0.1.h5
        pretrained_weights = '../ins/weights/mask_rcnn_resnet50_oid_v1.0.h5'
        dir_tfrecord       = '../ins-tfrecord'
        batch_size         = 2
    elif RUN_ON == 'kaggle':
        # Step 1: Get the credential from the Cloud SDK
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        user_credential = user_secrets.get_gcloud_credential()
        # Step 2: Set the credentials
        user_secrets.set_tensorflow_credential(user_credential)
        # Step 3: Use a familiar call to get the GCS path of the dataset
        from kaggle_datasets import KaggleDatasets
        GCS_DS_PATH = KaggleDatasets().get_gcs_path('instfrecord')

        dir_dataset        = 'gs://tyu-ins-sample'
        pretrained_weights = '../input/download-models/resnet50_oid_v1.0.1.h5'
        dir_tfrecord       = GCS_DS_PATH
        batch_size         = 16 * STRATEGY.num_replicas_in_sync if TPU else \
                             4  * STRATEGY.num_replicas_in_sync if GPU else \
                             4
    else:
        dir_dataset        = 'gs://tyu-ins-sample'
        pretrained_weights = './weights/resnet50_oid_v1.0.1.h5'
        dir_tfrecord       = 'gs://tyu-ins-sample-tfrecord'
        batch_size         = 16 * STRATEGY.num_replicas_in_sync if TPU else \
                             4  * STRATEGY.num_replicas_in_sync if GPU else \
                             4
    # Dir to save checkpoint.
    dir_snapshot = './keras_maskrcnn/bin/snapshot/'
    path_anno    = dir_dataset + '/annotation-instance-segmentation/'

    CLI_ARGS = [
        # '--snapshot', SNAPSHOT_DIR + 'mask_rcnn_resnet50_oid_v1.0.h5',
        # '--imagenet-weights',
        # '--freeze-backbone',
        '--weights', pretrained_weights,
        '--epochs', '2',
        '--gpu', '2',
        '--steps', '20',
        '--snapshot-path', dir_snapshot,
        '--lr', '1e-5',
        '--backbone', 'resnet50',
        '--group_method', 'random',
        '--batch-size', f'{batch_size}',
        'csv',
        dir_dataset,  # dataset_location or image_dir
        path_anno + f'train/challenge-2019-train-masks/challenge-2019-train-segmentation-masks.csv',
        path_anno + 'metadata/challenge-2019-classes-description-segmentable.csv',
        '--val-annotations',
        path_anno + 'validation/challenge-2019-validation-masks/challenge-2019-validation-segmentation-masks.csv',
    ]
    CONFIG = parser.parse_args(CLI_ARGS)

FIX_INPUT_H_W = None  # (600, 600)
# tf.keras.backend.clear_session()
# tf.config.optimizer.set_jit(True) # Enable XLA.
# _________________________________________________________________________________________________
# #### Configurations that do not need to change:
import pandas as pd
# class_names.columns: ['MID', 'class_name'], where 'MID' is 'label_name'
CLASS_NAME_DF = pd.read_csv(CONFIG.class_names, names=['MID', 'class_name'])
N_CLASS       = len(CLASS_NAME_DF)

DTYPES_GPU = {
    'image': tf.uint8,
    'masks': tf.uint8,
}

DTYPES_TPU = {
    'image': tf.float32,
    'masks': tf.float32,
}

DTYPES = argparse.Namespace(**DTYPES_GPU)

# _________________________________________________________________________________________________
# Program:
# noinspection PyUnresolvedReferences
import tensorflow as tf

import keras_retinanet.losses
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.utils.tf_version import check_tf_version
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.anchors import AnchorParameters, guess_shapes

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_maskrcnn import losses
from keras_maskrcnn import models
from albumentations import *
#
from keras_maskrcnn.utils.ins_utils import *
from keras_maskrcnn.models.resnet import resnet_maskrcnn

from keras_maskrcnn.utils.tf_dataset import make_fn_decoder, make_fn_to_model_input


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, n_class, weights, nms=True, freeze_backbone=False, class_specific_filter=True, anchor_params=None):
    modifier = freeze_model if freeze_backbone else None

    model = model_with_weights(
        resnet_maskrcnn(
            n_class,
            backbone=backbone_retinanet,
            input_shape=(*FIX_INPUT_H_W, 3) if FIX_INPUT_H_W is not None else (None, None, 3),
            nms=nms,
            class_specific_filter=class_specific_filter,
            modifier=modifier,
            anchor_params=anchor_params
            # use_tpu=TPU is not None,
        ), weights=weights, skip_mismatch=True)
    training_model   = model
    prediction_model = model

    # compile model
    opt = tf.keras.optimizers.Adam(lr=1e-5, )  # clipnorm=0.001)

    # compile model
    training_model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal(),
            'masks'         : losses.mask(),
        },
        optimizer=opt,
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    callbacks = []
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.9,
        patience = 3,
        verbose  = 1,
        mode     = 'auto',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    ))

    return callbacks


if __name__ == '__main__':
    if not os.path.isdir(CONFIG.snapshot_path):
        os.makedirs(CONFIG.snapshot_path)
    # create object that stores backbone information
    backbone = models.backbone(CONFIG.backbone)

    def _create_model():
        # create the model
        if CONFIG.snapshot is not None:
            print('Loading model {}, this may take a second...'.format(CONFIG.snapshot))
            model = models.load_model(CONFIG.snapshot, backbone_name=CONFIG.backbone)
            training_model = model
            prediction_model = model
        else:
            weights = CONFIG.weights
            # default to imagenet if nothing else is specified
            if weights is None and CONFIG.imagenet_weights:
                weights = backbone.download_imagenet()
            anchor_params = None
            print('Creating model, this may take a second...')
            model, training_model, prediction_model = create_models(
                backbone_retinanet='resnet50',
                n_class=N_CLASS,
                weights=weights,
                nms=True,
                freeze_backbone=CONFIG.freeze_backbone,
                class_specific_filter=CONFIG.class_specific_filter,
                anchor_params=anchor_params
            )
        # print model summary
        # model.summary()
        return model, training_model, prediction_model

    loss_fn_reg = keras_retinanet.losses.smooth_l1()
    loss_fn_cls = keras_retinanet.losses.focal()
    loss_fn_msk = losses.mask()
    optimizer   = tf.keras.optimizers.Adam(lr=1e-5, )

    with STRATEGY.scope():
        model, model_train, model_predict = _create_model()
        print('Learning rate: {}'.format(K.get_value(model.optimizer.lr)))
        if CONFIG.lr > 0.0:
            K.set_value(model.optimizer.lr, CONFIG.lr)
            print('Updated learning rate: {}'.format(K.get_value(model.optimizer.lr)))

        MID_TO_NO = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(CLASS_NAME_DF['MID'], CLASS_NAME_DF.index, value_dtype=tf.int32), -1)

        fnames = tf.io.matching_files(f'{dir_tfrecord}/train/*.tfrecord')
        fnames = tf.random.shuffle(fnames)
        ds_fnames = tf.data.Dataset.from_tensor_slices(fnames).repeat()
        # ds_raw_example = tf.data.TFRecordDataset(ds_fnames)
        ds_raw_example = ds_fnames.interleave(tf.data.TFRecordDataset)
        ds_raw_example = ds_raw_example.shuffle(buffer_size=100, reshuffle_each_iteration=False)
        ds_raw_example = ds_raw_example.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        ds_example = ds_raw_example.map(lambda x: tf.io.parse_single_sequence_example(
            serialized=x,
            context_features={
                'image_id' : tf.io.FixedLenFeature([], tf.string),
                'image_raw': tf.io.FixedLenFeature([], tf.string),
            },
            sequence_features={
                'li_mask_id'   : tf.io.FixedLenSequenceFeature([], tf.string),
                'li_mask_raw'  : tf.io.FixedLenSequenceFeature([], tf.string),
                'li_label_name': tf.io.FixedLenSequenceFeature([], tf.string),
                'li_box'       : tf.io.FixedLenSequenceFeature([4], tf.float32),
            }
        ))
        ds_decoded = ds_example.map(make_fn_decoder(CONFIG.class_names, image_h_w=(600, 600)))
        ds_batch   = ds_decoded.padded_batch(CONFIG.batch_size, drop_remainder=True)
        ds_input   = ds_batch.map(make_fn_to_model_input(N_CLASS))

        import datetime
        log_dir     = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                     # histogram_freq=1,
                                                     profile_batch='10, 15')

        # if 'keras_predict':
        #     ds_input = ds_input.map(lambda x, y: x)
        #     preds = model.predict(ds_input, callbacks=[tb_callback, ], steps=5, verbose=1)

        if 'keras_fit':
            initial_epoch = 0
            if CONFIG.snapshot is not None:
                initial_epoch = int((CONFIG.snapshot.split('_')[-1]).split('.')[0])

            # start training
            model_train.fit(
                x=ds_input,
                steps_per_epoch=CONFIG.steps,
                epochs=CONFIG.epochs,
                verbose=1,
                callbacks=[tb_callback],
                # max_queue_size=1,
                # initial_epoch=initial_epoch,
            )

        # if 'manual_train':
        #     @tf.function(experimental_compile=True)
        #     def train_step(images, labels):
        #         true_reg, true_cls, true_msk = labels
        #
        #         with tf.GradientTape() as tape:
        #             reg, cls, msk = model_train(images)
        #             loss_reg = loss_fn_cls(true_reg, reg)
        #             loss_cls = loss_fn_reg(true_cls, cls)
        #             loss_msk = loss_fn_msk(true_msk, msk)
        #             tf.print(loss_reg, loss_cls, loss_msk)
        #
        #             loss = tf.reduce_mean([loss_reg, loss_cls, loss_msk])
        #
        #         layer_variables = model_train.trainable_variables
        #         grads = tape.gradient(loss, layer_variables)
        #         optimizer.apply_gradients(zip(grads, layer_variables))
        #
        #     for x, y in ds_input:
        #         train_step(x, y)
        #         # print('infer '*10)
        #         # model_train(x)
        #         # print('after '*10)







# coding: utf-8


__author__ = 'Frank Jing'

# _________________________________________________________________________________________________
# Configurations:
import os

RUN_ON = 'kaggle' if os.path.exists('/kaggle') else 'local'

if RUN_ON == 'kaggle':
    DATASET_DIR = 'gs://tyu-ins-sample'
    PRETRAINED_WEIGHTS = '../input/download-models/resnet50_oid_v1.0.1.h5'
    ON_SAMPLES = False
else:
    os.chdir('D:/venv-tensorflow2/keras-maskrcnn')
    DATASET_DIR = '../open-images-dataset'
    PRETRAINED_WEIGHTS = '../ins/weights/mask_rcnn_resnet50_oid_v1.0.h5'
    ON_SAMPLES = True


BATCH_SIZE = 2
# Dir to save checkpoint.
SNAPSHOT_DIR = './snapshot/'

# _________________________________________________________________________________________________
# #### Configurations that do not need to change:
ANNO_PATH = DATASET_DIR + '/annotation-instance-segmentation/'

CLI_ARGS = [
    # '--snapshot', SNAPSHOT_DIR + 'mask_rcnn_resnet50_oid_v1.0.h5',
    # '--imagenet-weights',
    # '--freeze-backbone',
    '--weights', PRETRAINED_WEIGHTS,
    '--epochs', '2',
    '--gpu', '2',
    '--steps', '5',
    '--snapshot-path', SNAPSHOT_DIR,
    '--lr', '1e-5',
    '--backbone', 'resnet50',
    '--group_method', 'group_method',
    '--batch-size', f'{BATCH_SIZE}',
    'csv',
    DATASET_DIR,  # dataset_location or image_dir
    ANNO_PATH + f'train/challenge-2019-train-masks/challenge-2019-train-segmentation-masks.csv',
    ANNO_PATH + 'metadata/challenge-2019-classes-description-segmentable.csv',
    '--val-annotations',
    ANNO_PATH + 'validation/challenge-2019-validation-masks/challenge-2019-validation-segmentation-masks.csv',
]

# _________________________________________________________________________________________________
# Program:
import argparse

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

# Set your own project id here
PROJECT_ID = 'banded-splicer-259715'
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, args, freeze_backbone=False, class_specific_filter=True, anchor_params=None):
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
    opt = tf.keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)

    # compile model
    training_model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal(),
            'masks'         : losses.mask(),
        },
        optimizer=opt
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    callbacks = []
    #
    # # save the last prediction model
    # if args.snapshots:
    #     # ensure directory created first; otherwise h5py will error after epoch.
    #     os.makedirs(args.snapshot_path, exist_ok=True)
    #     checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #         os.path.join(
    #             args.snapshot_path,
    #             f'{args.backbone}_fold_{args.fold}_last.h5'
    #         ),
    #         verbose=1,
    #     )
    #     checkpoint = RedirectModel(checkpoint, model)
    #     callbacks.append(checkpoint)
    #
    # tensorboard_callback = None
    # if args.tensorboard_dir:
    #     tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #         log_dir=args.tensorboard_dir,
    #         histogram_freq=0,
    #         batch_size=args.batch_size,
    #         write_graph=True,
    #         write_grads=False,
    #         write_images=False,
    #         embeddings_freq=0,
    #         embeddings_layer_names=None,
    #         embeddings_metadata=None
    #     )
    #     callbacks.append(tensorboard_callback)
    #
    # # Calculate mAP
    # if args.evaluation and validation_generator:
    #     evaluation = Evaluate(validation_generator,
    #                           tensorboard=tensorboard_callback,
    #                           weighted_average=args.weighted_average,
    #                           save_map_path=args.snapshot_path + '/mask_rcnn_fold_{}.txt'.format(args.fold))
    #     evaluation = RedirectModel(evaluation, prediction_model)
    #     callbacks.append(evaluation)

    # # save prediction model with mAP
    # if args.snapshots:
    #     checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #         os.path.join(
    #             args.snapshot_path,
    #             f'{config.backbone}_fold_{config.fold}_mAP.h5'
    #         ),
    #         verbose=1,
    #         save_best_only=False,
    #         monitor="mAP",
    #         mode='max'
    #     )
    #     checkpoint = RedirectModel(checkpoint, prediction_model)
    #     callbacks.append(checkpoint)

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


# def transform_wrapper(image, masks, labels, bboxes):
#     aug = transform_generator(image=image, masks=masks, labels=labels, bboxes=bboxes)
#     return (aug['image'], aug['masks'], aug['labels'], aug['bboxes'])
#
#
# def tf_py_transform(image, masks, labels, bboxes):
#     image2, masks2, labels2, bboxes2 = tf.py_function(transform_wrapper,
#                                                       inp=[image, masks, labels, bboxes],
#                                                       Tout=[tf.uint8, tf.uint8, tf.float32, tf.float32])
#     return {'image': image2, 'masks': masks2, 'labels': labels2, 'bboxes': bboxes2}


def create_generators(args):
    train_generator = CSVGenerator(
        args.annotations,
        args.class_names,
        image_dir=os.path.join(args.dataset_dir, 'train'),
        transform_generator=transform_generator,
        batch_size=args.batch_size,
        config=args.config,
        image_min_side=800,
        image_max_side=1024,
        group_method=args.group_method,
        is_rle=False
    )

    if args.val_annotations:
        validation_generator = CSVGenerator(
            args.val_annotations,
            args.class_names,
            image_dir=os.path.join(args.dataset_dir, 'validation'),
            batch_size=args.batch_size,
            config=args.config,
            image_min_side=800,
            image_max_side=1024,
            group_method=args.group_method,
            is_rle=False
        )
    else:
        validation_generator = None

    return train_generator, validation_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    :param parsed_args: parser.parse_args()
    :return: parsed_args
    """

    return parsed_args


def parse_args(args):
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

    return check_args(parser.parse_args(args))


if '__file__' in dir():
    # config = parse_args(sys.argv[1:])
    config = parse_args(CLI_ARGS)
else:
    config = parse_args(CLI_ARGS)

if __name__ == '__main__':
    # make sure keras is the minimum required version
    check_tf_version()
    if not os.path.isdir(config.snapshot_path):
        os.mkdir(config.snapshot_path)

    # create object that stores backbone information
    backbone = models.backbone(config.backbone)

    # from keras_maskrcnn.models.resnet import resnet_maskrcnn
    # mrcnn = resnet_maskrcnn(
    #     300,
    #     nms=True,
    #     modifier=None,
    #     class_specific_filter=config.class_specific_filter,
    #     anchor_params=None,
    # )

    # create the generators
    train_generator, validation_generator = create_generators(config)

    # create the model
    if config.snapshot is not None:
        print('Loading model {}, this may take a second...'.format(config.snapshot))
        model            = models.load_model(config.snapshot, backbone_name=config.backbone)
        training_model   = model
        prediction_model = model
    else:
        weights = config.weights
        # default to imagenet if nothing else is specified
        if weights is None and config.imagenet_weights:
            weights = backbone.download_imagenet()

        anchor_params = None

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.maskrcnn,
            num_classes=train_generator.num_classes(),
            weights=weights,
            args=config,
            freeze_backbone=config.freeze_backbone,
            class_specific_filter=config.class_specific_filter,
            anchor_params=anchor_params
        )

    # print model summary
    model.summary()

    print('Learning rate: {}'.format(K.get_value(model.optimizer.lr)))
    if config.lr > 0.0:
        K.set_value(model.optimizer.lr, config.lr)
        print('Updated learning rate: {}'.format(K.get_value(model.optimizer.lr)))

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        config,
    )

    initial_epoch = 0
    if config.snapshot is not None:
        initial_epoch = int((config.snapshot.split('_')[-1]).split('.')[0])

    # start training
    training_model.fit(
        x=train_generator,
        steps_per_epoch=config.steps,
        epochs=config.epochs,
        verbose=1,
        callbacks=callbacks,
        max_queue_size=1,
        initial_epoch=initial_epoch,
    )

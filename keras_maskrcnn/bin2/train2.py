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
RUN_ON = 'local' if os.path.exists('C:/') else \
         'kaggle' if os.path.exists('/kaggle') else \
         'gcp'

try:  # Detect hardware, return appropriate distribution strategy
    import tensorflow as tf
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', TPU.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    # instantiate a distribution strategy
    STRATEGY = tf.distribute.experimental.TPUStrategy(TPU)
except ValueError:
    TPU      = None
    STRATEGY = tf.distribute.OneDeviceStrategy('/CPU:0')

if '--csv' in sys.argv:
    # if cli is used, set config by cli args.
    CONFIG = parser.parse_args()
else:
    if RUN_ON == 'local':
        os.chdir(r'D:\venv-tensorflow2\keras-maskrcnn')
        dir_dataset        = '../open-images-dataset'
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
        batch_size         = 4 if TPU is None else 16 * STRATEGY.num_replicas_in_sync
    else:
        dir_dataset        = 'gs://tyu-ins-sample'
        pretrained_weights = '../input/download-models/resnet50_oid_v1.0.1.h5'
        dir_tfrecord       = 'gs://tyu-ins-sample-tfrecord'
        batch_size         = 4 if TPU is None else 16 * STRATEGY.num_replicas_in_sync
    # Dir to save checkpoint.
    dir_snapshot = './keras_maskrcnn/bin/snapshot/'
    path_anno    = dir_dataset + '/annotation-instance-segmentation/'

    CLI_ARGS = [
        # '--snapshot', SNAPSHOT_DIR + 'mask_rcnn_resnet50_oid_v1.0.h5',
        # '--imagenet-weights',
        # '--freeze-backbone',
        '--weights', pretrained_weights,
        '--epochs', '10',
        '--gpu', '2',
        '--steps', '5',
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

FIX_SHAPE = (256, 256, 3)
# tf.keras.backend.clear_session()
# tf.config.optimizer.set_jit(True) # Enable XLA.
# _________________________________________________________________________________________________
# #### Configurations that do not need to change:
import pandas as pd
# class_names.columns: ['MID', 'class_name'], where 'MID' is 'label_name'
CLASS_NAME_DF = pd.read_csv(CONFIG.class_names, names=['MID', 'class_name'])
N_CLASS       = len(CLASS_NAME_DF)

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

def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


# @tf.function  # (experimental_compile=True)
def create_models(backbone_retinanet, num_classes, weights, args, freeze_backbone=False, class_specific_filter=True, anchor_params=None):
    modifier = freeze_model if freeze_backbone else None

    model = model_with_weights(
        resnet_maskrcnn(
            num_classes,
            backbone=backbone_retinanet,
            input_shape=FIX_SHAPE if FIX_SHAPE is not None else (None, None, 3),
            nms=True,
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
        optimizer=opt
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


if 'tf.data.Dataset':
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
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        },
        sequence_features={
            'li_mask_id'   : tf.io.FixedLenSequenceFeature([ ], tf.string),
            'li_mask_raw'  : tf.io.FixedLenSequenceFeature([ ], tf.string),
            'li_label_name': tf.io.FixedLenSequenceFeature([ ], tf.string),
            'li_box'       : tf.io.FixedLenSequenceFeature([4], tf.float32),
        }
    ))

    def _decode_example(ctx, seq):
        image  = tf.image.decode_jpeg(ctx['image_raw'], channels=3)  # tf.cast(, tf.float32) / 255.0
        if FIX_SHAPE is not None:
            image  = tf.image.resize(image, size=FIX_SHAPE[:2])
        h, w   = tf.unstack(tf.shape(image)[:2])
        masks  = tf.map_fn(lambda x: tf.image.decode_png(x, channels=1),
                          elems=seq['li_mask_raw'], dtype='uint8')
        masks  = tf.image.resize(masks, size=(h, w))
        # normalized box to absolute box.
        boxes  = seq['li_box'] * tf.convert_to_tensor([w, h, w, h], dtype=seq['li_box'].dtype)
        # MID name to No.
        labels = MID_TO_NO.lookup(seq['li_label_name'])

        return {'image'      : image,
                'boxes'      : boxes,
                'labels'     : labels,
                'masks'      : masks,
                'masks_shape': tf.shape(masks)}

    ds_batch = ds_example.map(_decode_example).padded_batch(CONFIG.batch_size, drop_remainder=True)

    # tmp = list(ds_batch.take(3))

    def batch_to_input(batch):
        bat_image       = batch['image']
        bat_boxes       = batch['boxes']
        bat_labels      = batch['labels']
        bat_masks       = batch['masks']
        bat_masks_shape = batch['masks_shape']

        # TODO: compute resize shape.

        bat_size, bat_n_anno, bat_h, bat_w, *_ = tf.unstack(tf.shape(bat_masks))
        anchors = get_anchors_for_shape(image_shape=(bat_h, bat_w))

        bat_reg, bat_cls = get_anchor_targets_batch(anchors, bat_boxes, bat_labels, N_CLASS)

        # bat_masks has shape: (batch size, max_annotations, 4 + 1 + 2 + padded_mask.size)
        # last dim schema: x1 + y1 + x2 + y2 + label + padded_width + padded_height + padded_mask(flatten image-dims)
        pos_0 = bat_boxes
        pos_4 = tf.cast(bat_labels[..., tf.newaxis], tf.float32)
        pos_5 = tf.broadcast_to(tf.cast(bat_w, tf.float32), shape=(bat_size, bat_n_anno, 1))
        pos_6 = tf.broadcast_to(tf.cast(bat_h, tf.float32), shape=(bat_size, bat_n_anno, 1))
        pos_7 = tf.reshape(bat_masks, shape=(bat_size, bat_n_anno, -1))

        bat_masks = tf.concat([pos_0, pos_4, pos_5, pos_6, pos_7], axis=-1)

        return bat_image, (bat_reg, bat_cls, bat_masks)


    def get_anchor_targets_batch(
            anchors,
            bat_boxes,
            bat_labels,
            n_classes,
            negative_overlap=0.4,
            positive_overlap=0.5
    ):
        iou              = compute_iou_batch(anchors, bat_boxes)
        # for every anchor, there's a annotated box who has max iou with this anchor. call it "max_box" here.
        max_box_iou      = tf.reduce_max(iou, axis=-1)
        # shape: [batch_size, n_anchor]
        positive_indices = max_box_iou > positive_overlap
        ignore_indices   = (max_box_iou > negative_overlap) & ~positive_indices
        max_box_indices  = tf.argmax(iou, axis=-1, output_type=tf.int32)
        # value 0: background; 1: foreground; -1: ignore
        flags = tf.zeros_like(ignore_indices, tf.float32) + \
                tf.cast(positive_indices, tf.float32) + \
                tf.cast(ignore_indices, tf.float32) * (-1)

        # regression final shape: [batch_size, n_anchor, 4 + 1]
        max_boxes  = tf.map_fn(lambda box_ind: tf.gather(*box_ind), (bat_boxes, max_box_indices), dtype=tf.float32)
        regression = bbox_transform(anchors[tf.newaxis, ...], max_boxes)
        regression = tf.concat([regression, flags[..., tf.newaxis]], axis=-1)

        # classification final shape: [batch_size, n_anchor, n_class + 1]
        anchor_labels  = tf.map_fn(lambda label_ind: tf.gather(*label_ind), (bat_labels, max_box_indices), dtype=tf.int32)
        classification = tf.one_hot(anchor_labels, depth=n_classes, dtype=tf.float32)
        classification = tf.concat([classification, flags[..., tf.newaxis]], axis=-1)

        # TODO: ignore annotations outside of image

        # TODO: intention.

        return regression, classification


    def compute_iou_batch(anchors, bat_boxes):
        """Return shape: [batch_size, n_anchors, n_boxes]."""
        # shape: [batch_size, n_anchors, n_boxes, 4]
        anchors_bc   = anchors[tf.newaxis, :, tf.newaxis, :]
        bat_boxes_bc = bat_boxes[:, tf.newaxis, :, :]
        # shape: [batch_size, n_anchor, n_boxes]
        area_anchors = (anchors_bc[:, :, :, 2] - anchors_bc[:, :, :, 0]) * \
                       (anchors_bc[:, :, :, 3] - anchors_bc[:, :, :, 1])
        area_boxes   = (bat_boxes_bc[:, :, :, 2] - bat_boxes_bc[:, :, :, 0]) * \
                       (bat_boxes_bc[:, :, :, 3] - bat_boxes_bc[:, :, :, 1])
        # shape: [batch_size, n_anchor, n_boxes]
        x1_intersect = tf.maximum(bat_boxes_bc[:, :, :, 0], anchors_bc[:, :, :, 0])
        y1_intersect = tf.maximum(bat_boxes_bc[:, :, :, 1], anchors_bc[:, :, :, 1])
        x2_intersect = tf.minimum(bat_boxes_bc[:, :, :, 2], anchors_bc[:, :, :, 2])
        y2_intersect = tf.minimum(bat_boxes_bc[:, :, :, 3], anchors_bc[:, :, :, 3])
        # shape: [batch_size, n_anchor, n_boxes]
        intersection = tf.clip_by_value((x2_intersect - x1_intersect), 0, np.inf) * \
                       tf.clip_by_value((y2_intersect - y1_intersect), 0, np.inf)
        union = area_anchors + area_boxes - intersection

        return intersection / union


    def bbox_transform(anchors, gt_boxes, mean=(0, 0, 0, 0), std=(0.2, 0.2, 0.2, 0.2)):
        """Compute bounding-box regression targets for an image. Refer: keras_retinanet.utils.anchors.bbox_transform"""
        mean = tf.constant(mean, dtype=tf.float32)
        std  = tf.constant(std,  dtype=tf.float32)
        w    = anchors[..., 2] - anchors[..., 0]
        h    = anchors[..., 3] - anchors[..., 1]

        delta_normal = (gt_boxes - anchors) / tf.stack([w, h, w, h], axis=-1)

        return (delta_normal - mean) / std


    def get_anchors_for_shape(image_shape,  # (height, width)
                              pyramid_levels=(3, 4, 5, 6, 7),
                              anchor_params=AnchorParameters.default,):

        def _get_centered_anchors(base_size, ratios, scales):
            # ratioed_boxes is centered boxes with specified aspect ratio and area of 1.
            ratioed_boxes = tf.map_fn(lambda r: tf.concat([[-1 / 2], [-r / 2], [1 / 2], [r / 2]], axis=0) / tf.sqrt(r),
                                      elems=ratios)
            # ratioed_scaled_boxes[i][j] is box that has ratio of ratios[i] and scale of scales[j].
            ratioed_scaled_boxes = base_size * ratioed_boxes[:, tf.newaxis, :] * scales[tf.newaxis, :, tf.newaxis]

            return tf.reshape(ratioed_scaled_boxes, (-1, 4))

        def _get_shifted_anchors(shape, stride, anchors):
            # shift_along_x[i] is anchors that shift along x by (i + 0.5) * stride element-wise.
            shift_along_x = tf.map_fn(
                lambda ix: anchors + tf.constant([1., 0., 1., 0.], shape=(1, 4)) * (ix + 0.5) * stride,
                elems=tf.range(shape[1], dtype='float32'))
            # shift_along_y_x[iy][ix] is anchors that shift along y by ((iy + 0.5) * stride)
            # and x by ((ix + 0.5) * stride) element-wise.
            shift_along_y_x = tf.map_fn(
                lambda iy: shift_along_x + tf.constant([0., 1., 0., 1.], shape=(1, 1, 4)) * (iy + 0.5) * stride,
                elems=tf.range(shape[0], dtype='float32'))

            return tf.reshape(shift_along_y_x, (-1, 4))

        image_shapes = [(tf.convert_to_tensor(image_shape) + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
        # compute anchors over all pyramid levels
        all_anchors = []
        for i_level, p in enumerate(pyramid_levels):
            anchors = _get_centered_anchors(base_size=anchor_params.sizes[i_level],
                                            ratios=anchor_params.ratios,
                                            scales=anchor_params.scales)
            shifted_anchors = _get_shifted_anchors(image_shapes[i_level], anchor_params.strides[i_level], anchors)
            all_anchors.append(shifted_anchors)

        return tf.concat(all_anchors, axis=0)


    ds_inp = ds_batch.map(batch_to_input)
    tt = list(ds_inp.take(3))


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
                num_classes=N_CLASS,
                weights=weights,
                args=CONFIG,
                freeze_backbone=CONFIG.freeze_backbone,
                class_specific_filter=CONFIG.class_specific_filter,
                anchor_params=anchor_params
            )
        # print model summary
        model.summary()
        return model, training_model, prediction_model

    with STRATEGY.scope():
        model, model_train, model_predict = _create_model()

    print('Learning rate: {}'.format(K.get_value(model.optimizer.lr)))
    if CONFIG.lr > 0.0:
        K.set_value(model.optimizer.lr, CONFIG.lr)
        print('Updated learning rate: {}'.format(K.get_value(model.optimizer.lr)))

    # # create the callbacks
    # callbacks = create_callbacks(
    #     model,
    #     model_train,
    #     model_predict,
    #     validation_generator,
    #     config,
    # )

    initial_epoch = 0
    if CONFIG.snapshot is not None:
        initial_epoch = int((CONFIG.snapshot.split('_')[-1]).split('.')[0])

    # start training
    model_train.fit(
        x=ds_inp,
        steps_per_epoch=CONFIG.steps,
        epochs=CONFIG.epochs,
        verbose=1,
        callbacks=None,
        max_queue_size=1,
        initial_epoch=initial_epoch,
    )

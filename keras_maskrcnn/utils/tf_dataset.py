
import numpy as np
import pandas as pd
import tensorflow as tf

from keras_retinanet.utils.anchors import AnchorParameters

# _________________________________________________________________________________________________
# Methods:
parse_single_sequence_example = lambda x: tf.io.parse_single_sequence_example(
    serialized=x,
    context_features={
        'image_id': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    },
    sequence_features={
        'li_mask_id': tf.io.FixedLenSequenceFeature([], tf.string),
        'li_mask_raw': tf.io.FixedLenSequenceFeature([], tf.string),
        'li_label_name': tf.io.FixedLenSequenceFeature([], tf.string),
        'li_box': tf.io.FixedLenSequenceFeature([4], tf.float32),
    }
)


def make_decoder(klass_csv_path, image_h_w=None, image_dtype=tf.uint8):
    """_decode_example need mid-to-no lookup table and image_h_w, so can't get decoder directly."""
    klass_df  = pd.read_csv(klass_csv_path, names=['MID', 'class_name'])
    mid_to_no = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(klass_df['MID'], klass_df.index, value_dtype=tf.int32), -1)
    if image_h_w is None:      # no change h, w
        compute_image_h_w = lambda h_w: h_w
    elif callable(image_h_w):  # compute h, w by given callable of image_h_w.
        compute_image_h_w = image_h_w
    else:                      # return fixed h, w specified by image_h_w.
        compute_image_h_w = lambda _: image_h_w

    def _decode_example(ctx, seq):
        image = tf.cast(tf.image.decode_jpeg(ctx['image_raw'], channels=3),
                        image_dtype)  # tf.cast(, tf.float32) / 255.0
        h, w = compute_image_h_w(tf.unstack(tf.shape(image)[:2]))
        image = tf.image.resize(image, size=(h, w))
        masks = tf.map_fn(lambda x: tf.image.decode_png(x, channels=1),
                          elems=seq['li_mask_raw'], dtype=image_dtype)
        masks = tf.image.resize(masks, size=(h, w))
        # normalized box to absolute box.
        boxes = seq['li_box'] * tf.convert_to_tensor([w, h, w, h], dtype=seq['li_box'].dtype)
        # MID name to No.
        labels = mid_to_no.lookup(seq['li_label_name'])

        return {'image_id': ctx['image_id'],
                'image': image,
                'mask_ids': seq['li_mask_id'],
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'masks_shape': tf.shape(masks)}

    return _decode_example


def to_model_input(batch):
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


if __name__ == '__main__':
    from keras_maskrcnn.bin2.train2 import *

    if not os.path.isdir(CONFIG.snapshot_path):
        os.makedirs(CONFIG.snapshot_path)

    with STRATEGY.scope():
        # model, training_model, prediction_model = create_models(
        #     backbone_retinanet='resnet50',
        #     n_class=N_CLASS,
        #     weights=CONFIG.weights,
        #     nms=True,
        #     freeze_backbone=CONFIG.freeze_backbone,
        #     class_specific_filter=CONFIG.class_specific_filter,
        #     anchor_params=None
        # )
        # print('Learning rate: {}'.format(K.get_value(model.optimizer.lr)))
        # if CONFIG.lr > 0.0:
        #     K.set_value(model.optimizer.lr, CONFIG.lr)
        #     print('Updated learning rate: {}'.format(K.get_value(model.optimizer.lr)))

        image_ids = ['83fe588b6e01ef7a',
                     '84c7775f391a85e4',
                     '8736925341819488',
                     '8746dc11f48a0d2c',
                     '879c97c0e6abd298',
                     '8836a53b50cb3e28',
                     '884d87c19c1bdf88',
                     '8952392367b3b61c']
        # image_ids = tf.convert_to_tensor(image_ids)

        fnames         = tf.io.matching_files(f'{dir_tfrecord}/train/*.tfrecord')
        fnames         = tf.random.shuffle(fnames)
        ds_fnames      = tf.data.Dataset.from_tensor_slices(fnames)  # .repeat()
        # ds_raw_example = tf.data.TFRecordDataset(ds_fnames)
        ds_raw_example = ds_fnames.interleave(tf.data.TFRecordDataset)
        ds_raw_example = ds_raw_example.shuffle(buffer_size=100, reshuffle_each_iteration=False)
        ds_raw_example = ds_raw_example.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        ds_example  = ds_raw_example.map(parse_single_sequence_example)
        ds_filtered = ds_example.filter(lambda ctx, seq: tf.reduce_any(tf.equal(ctx['image_id'], image_ids)))
        ds_decoded  = ds_filtered.map(make_decoder(CONFIG.class_names))


        ds_batch    = ds_decoded.padded_batch(4, drop_remainder=True)
        ds_input    = ds_batch.map(to_model_input)


        tt = list(ds_batch)









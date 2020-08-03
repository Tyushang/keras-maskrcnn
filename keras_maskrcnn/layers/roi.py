import tensorflow.keras.backend as K
from tensorflow.keras import layers
import keras_retinanet.backend

from .. import backend

# TODO: Remove this (necessary for a workaround).
import tensorflow as tf


class RoiAlign(layers.Layer):
    def __init__(self, crop_size=(14, 14), parallel_iterations=32, **kwargs):
        self.crop_size = crop_size
        self.parallel_iterations = parallel_iterations

        super(RoiAlign, self).__init__(**kwargs)

    def map_to_level(self, boxes, canonical_size=224, canonical_level=1, min_level=0, max_level=4):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        w = x2 - x1
        h = y2 - y1

        size = K.sqrt(w * h)

        levels = backend.floor(canonical_level + backend.log2(size / canonical_size + K.epsilon()))
        levels = K.clip(levels, min_level, max_level)

        return levels

    def call(self, inputs, **kwargs):
        image_shape = K.cast(inputs[0], K.floatx())
        boxes       = K.stop_gradient(inputs[1])
        scores      = K.stop_gradient(inputs[2])
        fpn         = [K.stop_gradient(i) for i in inputs[3:]]

        def _roi_align(args):
            boxes  = args[0]
            scores = args[1]
            fpn    = args[2]

            # compute from which level to get features from
            # BUG: not conform to FPN paper!
            target_levels = self.map_to_level(boxes)

            # process each pyramid independently
            rois           = []
            ordered_indices = []
            for i in range(len(fpn)):
                # select the boxes and classification from this pyramid level
                indices = keras_retinanet.backend.where(K.equal(target_levels, i))
                ordered_indices.append(indices)

                level_boxes = keras_retinanet.backend.gather_nd(boxes, indices)
                fpn_shape   = K.cast(K.shape(fpn[i]), dtype=K.floatx())

                # convert to expected format for crop_and_resize
                x1 = level_boxes[:, 0]
                y1 = level_boxes[:, 1]
                x2 = level_boxes[:, 2]
                y2 = level_boxes[:, 3]
                level_boxes = K.stack([
                    (y1 / image_shape[1] * fpn_shape[0]) / (fpn_shape[0] - 1),
                    (x1 / image_shape[2] * fpn_shape[1]) / (fpn_shape[1] - 1),
                    (y2 / image_shape[1] * fpn_shape[0] - 1) / (fpn_shape[0] - 1),
                    (x2 / image_shape[2] * fpn_shape[1] - 1) / (fpn_shape[1] - 1),
                ], axis=1)

                # append the rois to the list of rois
                rois.append(backend.crop_and_resize(
                    K.expand_dims(fpn[i], axis=0),
                    level_boxes,
                    tf.zeros((K.shape(level_boxes)[0],), dtype='int32'),  # TODO: Remove this workaround (https://github.com/tensorflow/tensorflow/issues/33787).
                    self.crop_size
                ))

            # concatenate rois to one blob
            rois = K.concatenate(rois, axis=0)

            # reorder rois back to original order
            indices = K.concatenate(ordered_indices, axis=0)
            rois    = keras_retinanet.backend.scatter_nd(indices, rois, K.cast(K.shape(rois), 'int64'))

            return rois

        roi_batch = keras_retinanet.backend.map_fn(
            _roi_align,
            elems=[boxes, scores, fpn],
            dtype=K.floatx(),
            parallel_iterations=self.parallel_iterations
        )

        return roi_batch

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], None, self.crop_size[0], self.crop_size[1], input_shape[3][-1])

    def get_config(self):
        config = super(RoiAlign, self).get_config()
        config.update({
            'crop_size' : self.crop_size,
        })

        return config


class RoiAlignTPU(layers.Layer):
    def __init__(self, crop_size=(14, 14), parallel_iterations=32, **kwargs):
        self.crop_size = crop_size
        self.parallel_iterations = parallel_iterations

        super(RoiAlignTPU, self).__init__(**kwargs)

    def map_to_level(self, boxes, canonical_size=224, canonical_level=4, min_level=3, max_level=7):
        """See: FPN paper(https://arxiv.org/pdf/1612.03144.pdf) Equation (1)."""
        x1, y1, x2, y2 = tf.unstack(boxes, axis=-1)
        # tf.print(boxes)

        w = x2 - x1
        h = y2 - y1

        levels = backend.floor(canonical_level + backend.log2(K.sqrt(w * h) / canonical_size + K.epsilon()))
        levels = K.clip(levels, min_level, max_level)

        return levels

    def call(self, inputs, **kwargs):
        image_shape   = inputs[0]  # K.cast(inputs[0], K.floatx())
        bat_boxes_abs = inputs[1]  # K.stop_gradient(inputs[1])
        bat_scores    = inputs[2]  # K.stop_gradient(inputs[2])
        bat_features  = inputs[3:]  # [K.stop_gradient(i) for i in inputs[3:]]

        if 'check_boxes':
            b, h, w, *_ = tf.unstack(image_shape, name='tyu_ra_unstack137')
            # use broadcast to clip on last dimension.
            bat_boxes_abs  = tf.clip_by_value(bat_boxes_abs,
                                              clip_value_min=0.0,
                                              clip_value_max=tf.convert_to_tensor([w, h, w, h], dtype=tf.float32) - 1.0)
            # x1 must lt x2, and y1 must lt y2.
            x1, y1, x2, y2 = tf.unstack(bat_boxes_abs, axis=-1)
            bat_boxes_abs  = tf.clip_by_value(bat_boxes_abs,
                                              clip_value_min=0.0,
                                              clip_value_max=tf.stack([x2, y2, x2, y2], axis=-1))

        def _roi_align(args):
            crop_h, crop_w = self.crop_size
            # boxes_abs shape: [max_detections, 4]
            # scores    shape: [max_detections, ]
            # features  shape: [H, W, C] for every feature in ['P3', 'P4', 'P5', 'P6', 'P7']
            boxes_abs, scores, features = args
            max_detections, *_ = tf.unstack(tf.shape(boxes_abs), name='tyu_ra_unstack154')
            max_feat_height, max_feat_width, *_ = tf.unstack(tf.shape(features[0]), name='tyu_ra_unstack155')
            feat_shape_table = tf.stack(list(map(lambda f: tf.shape(f), features)), axis=0)

            # compute from which level to get features from
            # shape: [max_detections, ], after-same.
            levels       = self.map_to_level(boxes_abs)
            feat_indices = tf.cast(levels - 3, tf.int32)

            # shape: [max_detections, 2]
            boundaries       = tf.map_fn(lambda i: feat_shape_table[i][:2], feat_indices) - 1
            bound_y, bound_x = tf.unstack(boundaries, axis=-1)

            padded_feats = []
            for f in features:
                padded_feats.append(
                    tf.image.pad_to_bounding_box(f, 0, 0, max_feat_height, max_feat_width))
            # shape: [n_feature, max_height, max_width, C]
            padded_feats = tf.stack(padded_feats, axis=0)

            # convert absolute boxes to level boxes.
            # shape: [max_detections, ], after-same.
            scales         = tf.math.pow(2.0, levels)
            boxes_level    = boxes_abs / scales[:, tf.newaxis]
            x1, y1, x2, y2 = tf.unstack(boxes_level, axis=-1)
            h              = y2 - y1
            w              = x2 - x1

            # gather nearby points.
            # shape: [max_detections, crop_h or crop_w]
            box_grid_y = (tf.range(crop_h, dtype=h.dtype)[tf.newaxis, :] + 0.5) * (h / crop_h)[:, tf.newaxis] \
                         + y1[:, tf.newaxis]
            box_grid_x = (tf.range(crop_w, dtype=w.dtype)[tf.newaxis, :] + 0.5) * (w / crop_w)[:, tf.newaxis] \
                         + x1[:, tf.newaxis]

            # shape: [max_detections, crop_h]
            index_grid_y0 = tf.clip_by_value(tf.cast(tf.floor(box_grid_y), tf.int32),
                                             clip_value_min=0, clip_value_max=bound_y[:, tf.newaxis])
            index_grid_y1 = tf.clip_by_value(index_grid_y0 + 1,
                                             clip_value_min=0, clip_value_max=bound_y[:, tf.newaxis])
            # shape: [max_detections, 2 * crop_h]
            index_grid_y  = tf.reshape(tf.stack([index_grid_y0, index_grid_y1], axis=-1),
                                       shape=(max_detections, 2 * crop_h))
            # shape: [max_detections, crop_w]
            index_grid_x0 = tf.clip_by_value(tf.cast(tf.floor(box_grid_x), tf.int32),
                                             clip_value_min=0, clip_value_max=bound_x[:, tf.newaxis])
            index_grid_x1 = tf.clip_by_value(index_grid_x0 + 1,
                                             clip_value_min=0, clip_value_max=bound_x[:, tf.newaxis])
            # shape: [max_detections, 2 * crop_w]
            index_grid_x  = tf.reshape(tf.stack([index_grid_x0, index_grid_x1], axis=-1),
                                       shape=(max_detections, 2 * crop_w))
            # shape: [max_detections, 2 * crop_h, 2 * crop_w, 2]
            index_grid = tf.cast(tf.stack([
                tf.broadcast_to(index_grid_y[:, :, tf.newaxis], shape=(max_detections, 2 * crop_h, 2 * crop_w)),
                tf.broadcast_to(index_grid_x[:, tf.newaxis, :], shape=(max_detections, 2 * crop_h, 2 * crop_w))
            ], axis=-1), dtype=tf.int32)
            # shape: [max_detections, 2 * crop_h, 2 * crop_w, C]
            nearby_points = tf.map_fn(lambda args: tf.gather_nd(padded_feats[args[0]], args[1]),
                                      elems=(feat_indices, index_grid),
                                      dtype=padded_feats.dtype)

            # The RoIAlign feature f can be computed by bilinear interpolation of four
            # neighboring feature points f0, f1, f2, and f3.
            # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
            #                       [f10, f11]]
            # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
            # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
            # here, we 1st multiply nearby-points by interpolation kernel, then apply avg_pool to get result.
            # shape: [max_detections, crop_h], after-same.
            ly = box_grid_y - tf.cast(index_grid_y0, dtype=box_grid_y.dtype)
            hy = 1.0 - ly
            # shape: [max_detections, 2 * crop_h, 1]
            kernel_y = tf.reshape(tf.stack([hy, ly], axis=-1), [max_detections, 2 * crop_h, 1])
            # shape: [max_detections, crop_w], after-same.
            lx = box_grid_x - tf.cast(index_grid_x0, dtype=box_grid_x.dtype)
            hx = 1.0 - lx
            # shape: [max_detections, 2 * crop_w]
            kernel_x = tf.reshape(tf.stack([hx, lx], axis=-1), [max_detections, 1, 2 * crop_w])
            # Use implicit broadcast to generate the interpolation kernel. The
            # multiplier `4` is for avg pooling.
            # shape: [max_detections, 2 * crop_h, 2 * crop_h]
            interpolation_kernel = kernel_y * kernel_x * 4

            weighted  = nearby_points * interpolation_kernel[:, :, :, tf.newaxis]
            # shape: [max_detections, crop_h, crop_h, C]
            box_feats = tf.nn.avg_pool(weighted, ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1], padding='VALID')

            return box_feats

        roi_batch = keras_retinanet.backend.map_fn(
            _roi_align,
            elems=(bat_boxes_abs, bat_scores, bat_features),
            dtype=K.floatx(),
            parallel_iterations=self.parallel_iterations
        )

        return roi_batch

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], None, self.crop_size[0], self.crop_size[1], input_shape[3][-1])

    def get_config(self):
        config = super(RoiAlignTPU, self).get_config()
        config.update({
            'crop_size' : self.crop_size,
        })

        return config


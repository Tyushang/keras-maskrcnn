import tensorflow.keras.backend as K
import tf_retinanet.backend
from . import backend


def mask(iou_threshold=0.5, mask_size=(28, 28), parallel_iterations=32):
    def _mask_conditional(y_true, y_pred):
        # if there are no masks annotations, return 0; else, compute the masks loss
        loss = backend.cond(
            K.any(K.equal(K.shape(y_true), 0)),
            lambda: K.cast_to_floatx(0.0),
            lambda: _mask_batch(y_true, y_pred, iou_threshold=iou_threshold, mask_size=mask_size, parallel_iterations=parallel_iterations)
        )
        return loss

    def _mask_batch(y_true, y_pred, iou_threshold=0.5, mask_size=(28, 28), parallel_iterations=32):
        # split up the different predicted blobs
        boxes = y_pred[:, :, :4]
        masks = y_pred[:, :, 4:]

        # split up the different blobs
        annotations  = y_true[:, :, :5]
        width        = K.cast(y_true[0, 0, 5], dtype='int32')
        height       = K.cast(y_true[0, 0, 6], dtype='int32')
        masks_target = y_true[:, :, 7:]

        # reshape the masks back to their original size
        masks_target = K.reshape(masks_target, (K.shape(masks_target)[0], K.shape(masks_target)[1], height, width))
        masks        = K.reshape(masks, (K.shape(masks)[0], K.shape(masks)[1], mask_size[0], mask_size[1], -1))

        def _mask(args):
            boxes = args[0]
            masks = args[1]
            annotations = args[2]
            masks_target = args[3]

            return compute_mask_loss(
                boxes,
                masks,
                annotations,
                masks_target,
                width,
                height,
                iou_threshold = iou_threshold,
                mask_size     = mask_size,
            )

        mask_batch_loss = tf_retinanet.backend.map_fn(
            _mask,
            elems=[boxes, masks, annotations, masks_target],
            dtype=K.floatx(),
            parallel_iterations=parallel_iterations
        )

        return K.mean(mask_batch_loss)

    return _mask_conditional


def compute_mask_loss(
    boxes,
    masks,
    annotations,
    masks_target,
    width,
    height,
    iou_threshold=0.5,
    mask_size=(28, 28)
):
    # compute overlap of boxes with annotations
    iou                  = backend.overlap(boxes, annotations)
    argmax_overlaps_inds = K.argmax(iou, axis=1)
    max_iou              = K.max(iou, axis=1)

    # filter those with IoU > 0.5
    indices              = tf_retinanet.backend.where(K.greater_equal(max_iou, iou_threshold))
    boxes                = tf_retinanet.backend.gather_nd(boxes, indices)
    masks                = tf_retinanet.backend.gather_nd(masks, indices)
    argmax_overlaps_inds = K.cast(tf_retinanet.backend.gather_nd(argmax_overlaps_inds, indices), 'int32')
    labels               = K.cast(K.gather(annotations[:, 4], argmax_overlaps_inds), 'int32')

    # make normalized boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    boxes = K.stack([
        y1 / (K.cast(height, dtype=K.floatx()) - 1),
        x1 / (K.cast(width, dtype=K.floatx()) - 1),
        (y2 - 1) / (K.cast(height, dtype=K.floatx()) - 1),
        (x2 - 1) / (K.cast(width, dtype=K.floatx()) - 1),
    ], axis=1)

    # crop and resize masks_target
    masks_target = K.expand_dims(masks_target, axis=3)  # append a fake channel dimension
    masks_target = backend.crop_and_resize(
        masks_target,
        boxes,
        argmax_overlaps_inds,
        mask_size
    )
    masks_target = masks_target[:, :, :, 0]  # remove fake channel dimension

    # gather the predicted masks using the annotation label
    masks = backend.transpose(masks, (0, 3, 1, 2))
    label_indices = K.stack([
        K.arange(K.shape(labels)[0]),
        labels
    ], axis=1)
    masks = tf_retinanet.backend.gather_nd(masks, label_indices)

    # compute mask loss
    mask_loss  = K.binary_crossentropy(masks_target, masks)
    normalizer = K.shape(masks)[0] * K.shape(masks)[1] * K.shape(masks)[2]
    normalizer = K.maximum(K.cast(normalizer, K.floatx()), 1)
    mask_loss  = K.sum(mask_loss) / normalizer

    return mask_loss

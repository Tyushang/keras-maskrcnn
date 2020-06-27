#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers

import tf_retinanet.layers
import tf_retinanet.models.retinanet

from ..layers.misc import Shape, ConcatenateBoxes, Cast
from ..layers.roi import RoiAlign
from ..layers.upsample import Upsample


def default_mask_model(
    num_classes,
    pyramid_feature_size=256,
    mask_feature_size=256,
    roi_size=(14, 14),
    mask_size=(28, 28),
    name='mask_submodel',
    mask_dtype=K.floatx(),
    retinanet_dtype=K.floatx()
):

    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros',
        'activation'         : 'relu',
    }

    inputs  = tf.keras.layers.Input(shape=(None, roi_size[0], roi_size[1], pyramid_feature_size))
    outputs = inputs

    # casting to the desidered data type, which may be different than
    # the one used for the underlying keras-retinanet model
    if mask_dtype != retinanet_dtype:
        outputs = tf.keras.layers.TimeDistributed(
            Cast(dtype=mask_dtype),
            name='cast_masks')(outputs)

    for i in range(4):
        outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
            filters=mask_feature_size,
            **options
        ), name='roi_mask_{}'.format(i))(outputs)

    # perform upsampling + conv instead of deconv as in the paper
    # https://distill.pub/2016/deconv-checkerboard/
    outputs = tf.keras.layers.TimeDistributed(
        Upsample(mask_size),
        name='roi_mask_upsample')(outputs)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
        filters=mask_feature_size,
        **options
    ), name='roi_mask_features')(outputs)

    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        activation='sigmoid'
    ), name='roi_mask')(outputs)

    # casting back to the underlying keras-retinanet model data type
    if mask_dtype != retinanet_dtype:
        outputs = tf.keras.layers.TimeDistributed(
            Cast(dtype=retinanet_dtype),
            name='recast_masks')(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_roi_submodels(num_classes, mask_dtype=K.floatx(), retinanet_dtype=K.floatx()):
    return [
        ('masks', default_mask_model(num_classes, mask_dtype=mask_dtype, retinanet_dtype=retinanet_dtype)),
    ]


def retinanet(input_shape, backbone, modifier_resnet):

    inputs = layers.Input(shape=(None, None, 3), name='image')

    # create the resnet backbone
    if backbone == 'resnet50':
        resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet101':
        resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet152':
        resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)
    else:
        raise Exception("Invalid Backbone!")
    # invoke modifier if given
    if modifier_resnet:
        resnet = modifier_resnet(resnet)

    x = resnet(inputs)

    # backbone_retinanet(
    #     num_classes,
    #     nms=True,
    #     class_specific_filter=class_specific_filter,
    #     modifier=modifier,
    #     anchor_params=anchor_params
    # )
    # def retinanet(
    #         inputs,
    #         backbone_layers,
    #         submodels,
    #         num_anchors=None,
    #         create_pyramid_features=fpn.create_pyramid_features,
    #         name='retinanet'
    # ):
    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)

    retinanet_model = tf_retinanet.models.retinanet.retinanet(
        inputs=inputs,
        backbone_layers=x[1:],
        num_classes=num_classes,
        num_anchors=anchor_params.num_anchors(),
    )

    # create the full model
    model = retinanet.retinanet_mask(inputs=inputs, num_classes=num_classes, backbone_layers=resnet.outputs[1:], mask_dtype=mask_dtype, **kwargs)



def create_maskrcnn(
    inputs,
    num_classes,
    retinanet_model,
    anchor_params=None,
    nms=True,
    class_specific_filter=True,
    name='retinanet_backboned_maskrcnn',
    roi_submodels=None,
    modifier=None,
):
    """ Construct a RetinaNet mask model on top of a retinanet bbox model.

    This model uses the retinanet bbox model and appends a few layers to compute masks.

    # Arguments
        inputs                : List of tf.keras.layers.Input. The first input is the image, the second input the blob of masks.
        num_classes           : Number of classes to classify.
        retinanet_model       : tf_retinanet.models.retinanet model, returning regression and classification values.
        anchor_params         : Struct containing anchor parameters. If None, default values are used.
        nms                   : Use NMS.
        class_specific_filter : Use class specific filtering.
        roi_submodels         : Submodels for processing ROIs.
        mask_dtype            : Data type of the masks, can be different from the main one.
        modifier              : Modifier for the underlying retinanet model, such as freeze.
        name                  : Name of the model.
        **kwargs              : Additional kwargs to pass to the retinanet bbox model.
    # Returns
        Model with inputs as input and as output the output of each submodel for each pyramid level and the detections.

        The order is as defined in submodels.
        ```
        [
            regression, classification, other[0], other[1], ..., boxes_masks, boxes, scores, labels, masks, other[0], other[1], ...
        ]
        ```
    """
    if anchor_params is None:
        anchor_params = tf_retinanet.utils.anchors.AnchorParameters.default

    image = inputs
    image_shape = Shape()(image)


    if modifier:
        retinanet_model = modifier(retinanet_model)

    # parse outputs
    regression     = retinanet_model.outputs[0]
    classification = retinanet_model.outputs[1]
    other          = retinanet_model.outputs[2:]
    features       = [retinanet_model.get_layer(name).output for name in ['P3', 'P4', 'P5', 'P6', 'P7']]



    # build boxes
    anchors = tf_retinanet.models.retinanet.build_anchors(anchor_params, image, features)
    boxes = tf_retinanet.layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = tf_retinanet.layers.ClipBoxes(name='clipped_boxes')([image, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = tf_retinanet.layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        max_detections        = 100,
        name                  = 'filtered_detections'
    )([boxes, classification] + other)

    # split up in known outputs and "other"
    boxes  = detections[0]
    scores = detections[1]

    # get the region of interest features
    rois = RoiAlign()([image_shape, boxes, scores] + features)

    # execute maskrcnn submodels
    maskrcnn_outputs = [submodel(rois) for _, submodel in roi_submodels]

    # concatenate boxes for loss computation
    trainable_outputs = [ConcatenateBoxes(name=name)([boxes, output]) for (name, _), output in zip(roi_submodels, maskrcnn_outputs)]

    # reconstruct the new output
    outputs = [regression, classification] + other + trainable_outputs + detections + maskrcnn_outputs

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)


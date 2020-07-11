#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"



import numpy as np


mask_shape = [28, 28]


def mask_loss_one_elem(y_true: np.ndarray, y_pred: np.ndarray, iou_threshold=0.5, mask_shape=(28, 28)):
    boxes_pred = y_pred[:, :4]
    masks_pred = y_pred[:, 4:]

    boxes_true = y_true[:, :4]
    label_true = y_true[:, 4]
    masks_true = y_true[:, 7:]
    width      = y_true[0, 5]
    height     = y_true[0, 6]

    masks_true = masks_true.reshape([masks_true.shape[0], height, width])
    masks_pred = masks_pred.reshape([masks_pred.shape[0], mask_shape[0], mask_shape[1], -1])


def mask_loss(y_true: np.ndarray, y_pred: np.ndarray):
    if y_true.size == 0:
        return 0.0

    boxes_pred = y_pred[:, :, :4]
    masks_pred = y_pred[:, :, 4:]

    boxes_true = y_true[:, :, :4]
    label_true = y_true[:, :, 4]
    masks_true = y_true[:, :, 7:]
    width      = y_true[0, 0, 5]
    height     = y_true[0, 0, 6]

    masks_true = masks_true.reshape([masks_true.shape[0], masks_true.shape[1], height, width])
    masks_pred = masks_pred.reshape([masks_pred.shape[0], masks_pred.shape[1], mask_shape[0], mask_shape[1], -1])

    map()

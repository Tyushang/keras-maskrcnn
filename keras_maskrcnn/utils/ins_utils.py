#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"


# _________________________________________________________________________________________________
# coding: utf-8
# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os.path
import random
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras_maskrcnn.preprocessing.generator import Generator


def get_image_size(path):
    # im = Image.open(image_path)
    # w, h = im.size
    # return w, h
    img = read_single_image(path)
    w, h, *_ = img.shape
    return w, h


def read_single_image(path):
    # img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    # return img
    return tf.image.decode_jpeg(tf.io.read_file(path)).numpy()


def rle_decode(mask_rle, shape=(1024, 1024)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def read_oid_segmentation_annotations(image_dir, mask_dir, csv_path, label_names):
    result = {}
    start_time = time.time()
    print('Reading anno {}'.format(csv_path))
    seg_df = pd.read_csv(csv_path, usecols=[
        'MaskPath', 'ImageID', 'LabelName', 'BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax'])
    # 'MaskPath' here actually is mask file name(e.g. 88e582a7b14e34a8_m039xj__6133896f.png)
    seg_df.rename({'MaskPath': 'MaskFilename'})
    for _, row in seg_df.iterrows():
        mask_filename, image_id, label_name, x1, x2, y1, y2 = row.to_list()
        x1, x2, y1, y2 = check_segmentation_annos(label_name, label_names, row, x1, x2, y1, y2)

        img_path  = os.path.join(image_dir, f'{image_id}.jpg')
        mask_path = os.path.join(mask_dir,  mask_filename)

        if img_path not in result:
            result[img_path] = []
        result[img_path].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'label_name': label_name, 'mask_path': mask_path})
    print('Total images: {} Reading time: {:.2f} sec'.format(len(result), time.time() - start_time))
    return result


def check_segmentation_annos(label_name, label_names, row, x1, x2, y1, y2):
    # Check that the bounding box is valid.
    if x1 < 0:
        # raise ValueError('line {}: negative x1 ({})'.format(i, x1))
        print('line {}: negative x1 ({})'.format(row, x1))
        x1 = 0
    if y1 < 0:
        # raise ValueError('line {}: negative y1 ({})'.format(i, y1))
        print('line {}: negative y1 ({})'.format(row, y1))
        y1 = 0
    if x2 > 1:
        # raise ValueError('line {}: invalid x2 ({})'.format(i, x2))
        print('line {}: invalid x2 ({})'.format(row, x2))
        x2 = 1
    if y2 > 1:
        # raise ValueError('line {}: invalid y2 ({})'.format(i, y2))
        print('line {}: invalid y2 ({})'.format(row, y2))
        y2 = 1
    if x2 <= x1:
        raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(row, x2, x1))
    if y2 <= y1:
        raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(row, y2, y1))
    if label_name not in label_names:
        raise ValueError('line {}: unknown class name: \'{}\' (mid_to_no: {})'.format(row, label_name, label_names))
    return x1, x2, y1, y2


def get_class_no_to_image_paths(mid_to_no, image_anno_desc):
    no_to_paths = {}
    for image_path, anno_list in image_anno_desc.items():
        for anno in anno_list:
            no = mid_to_no[anno['label_name']]
            if no not in no_to_paths:
                no_to_paths[no] = [image_path, ]
            else:
                no_to_paths[no].append(image_path)

    return no_to_paths


class CSVGenerator(Generator):
    def __init__(
        self,
        segmentations_csv,
        class_names_csv,
        image_dir,
        mask_dir=None,  # base dir of segmentations_csv
        is_rle=False,
        **kwargs
    ):
        self.image_dir      = image_dir
        self.image_paths    = []
        self.anno_descs     = {}
        self.mask_dir       = mask_dir
        self.is_rle         = is_rle

        # Take mask_dir from anno file if not explicitly specified.
        if self.mask_dir is None:
            self.mask_dir = os.path.dirname(segmentations_csv)

        # class_names.columns: ['No', 'MID', 'class_name'], where 'MID' is 'label_name'
        self.class_names: pd.DataFrame = pd.read_csv(class_names_csv, names=['MID', 'class_name'])\
            .rename_axis(index='No').reset_index()
        self.mid_to_no = self.class_names.set_index('MID')['No'].to_dict()
        self.no_to_mid = self.class_names.set_index('No')['MID'].to_dict()

        # csv with MaskPath,LabelName,BoxID,BoxXMin,BoxXMax,BoxYMin,BoxYMax
        self.anno_descs = read_oid_segmentation_annotations(
            self.image_dir, self.mask_dir, segmentations_csv, self.class_names['MID'].to_list())
        self.image_paths      = list(self.anno_descs.keys())

        self.idx_to_image_path = dict([(i, path) for i, path in enumerate(self.image_paths)])
        self.image_path_to_idx = dict([(path, i) for i, path in enumerate(self.image_paths)])
        self.class_no_to_image_paths = get_class_no_to_image_paths(self.mid_to_no, self.anno_descs)

        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.image_paths)

    def num_classes(self):
        return len(self.class_names)

    def name_to_label(self, name):
        return self.mid_to_no[name]

    def label_to_name(self, label):
        return self.no_to_mid[label]

    def image_path(self, image_index):
        return self.image_paths[image_index]
        # return os.path.join(self.mask_dir, self.image_paths[image_index])

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        return read_single_image(self.image_path(image_index))

    def load_annotations(self, image_index):
        path = self.image_paths[image_index]
        anno_list = self.anno_descs[path]

        annotations     = {
            # labels actually is class No. of corresponding bbox.
            'labels': np.empty((len(anno_list),)),
            'bboxes': np.empty((len(anno_list), 4)),
            'masks': [],
        }

        for idx, anno in enumerate(anno_list):
            if self.is_rle is False:
                mask = cv2.imread(anno['mask_path'], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print('Invalid mask: {}'.format(anno['mask_path']))
                    w, h = get_image_size(path)
                    # mask = np.zeros((h, w), dtype=np.uint8)
                    mask = tf.zeros((w, h), dtype='uint8')
            else:
                mask = rle_decode(anno['mask_path'], (1200, 1200))

            annotations['bboxes'][idx, 0] = float(anno['x1'] * mask.shape[1])  # = float(anno['x1']) ?
            annotations['bboxes'][idx, 1] = float(anno['y1'] * mask.shape[0])  # = float(anno['y1']) ?
            annotations['bboxes'][idx, 2] = float(anno['x2'] * mask.shape[1])  # = float(anno['x2']) ?
            annotations['bboxes'][idx, 3] = float(anno['y2'] * mask.shape[0])  # = float(anno['y2']) ?
            annotations['labels'][idx] = self.mid_to_no[anno['label_name']]

            mask = tf.cast(mask > 0, 'uint8')  # convert from 0-255 to binary mask
            annotations['masks'].append(tf.expand_dims(mask, axis=-1))

        return annotations

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its anno.
        """

        # randomly transform image and anno
        image, annotations = self.random_transform_group_entry(image, annotations)

        # preprocess the image
        image = self.preprocess_image(image)

        # resize image
        image, image_scale = self.resize_image(image)

        # resize masks
        # for i in range(len(annotations['masks'])):
        #     annotations['masks'][i], _ = self.resize_image(annotations['masks'][i])

        annotations['masks'] = list(map(
            lambda arr: self.resize_image(arr)[0], annotations['masks']
        ))

        # apply resizing to anno too
        annotations['bboxes'] *= image_scale

        return image, annotations

    def random_transform_group_entry(self, image, anno, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and anno
        # show_image(image)
        # print(image.min(), image.max())
        # print(anno)

        if self.transform_generator and len(anno['masks']) > 0:
            augmented = self.transform_generator(
                image=image,
                masks=np.stack(anno['masks'])[:, :, :, 0],
                labels=anno['labels'],
                bboxes=anno['bboxes'],
            )
            image = augmented['image']
            anno['masks']  = list(map(lambda arr: np.expand_dims(arr, -1), augmented['masks']))
            anno['bboxes'] = np.stack(augmented['bboxes'])

        return image, anno

    def group_images(self):
        print('Group images. Method: {}...'.format(self.group_method))

        if self.group_method == 'random_classes':
            class_no_list = list(self.class_no_to_image_paths.keys())
            self.groups = []
            from math import ceil
            n_step = ceil(self.size() / self.batch_size)
            for _ in range(n_step):
                batch_list = []
                for _ in range(self.batch_size):
                    rand_class_no   = random.choice(class_no_list)
                    rand_image_path = random.choice(self.class_no_to_image_paths[rand_class_no])
                    rand_image_idx  = self.image_path_to_idx[rand_image_path]
                    batch_list.append(rand_image_idx)
                self.groups.append(batch_list.copy())
            return

        # determine the order of the images
        indexes = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(indexes)
        elif self.group_method == 'ratio':
            indexes.sort(key=lambda x: self.image_aspect_ratio(x))
        # divide into groups, one group = one batch
        self.groups = [[indexes[i % len(indexes)] for i in range(start, start + self.batch_size)]
                       for start in range(0, len(indexes), self.batch_size)]

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their anno.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        # copy all anno / masks to the batch
        max_annotations = max(len(a['masks']) for a in annotations_group)
        # masks_batch has shape: (batch size, max_annotations, bbox_x1 + bbox_y1 + bbox_x2 + bbox_y2 + label + width + height + max_image_dimension)
        masks_batch = np.zeros((self.batch_size, max_annotations, 5 + 2 + max_shape[0] * max_shape[1]), dtype=K.floatx())
        for index, annotations in enumerate(annotations_group):
            try:
                masks_batch[index, :annotations['bboxes'].shape[0], :4] = annotations['bboxes']
            except:
                print('Error in compute targets!')
                print(index, annotations_group)

            masks_batch[index, :annotations['labels'].shape[0], 4] = annotations['labels']
            masks_batch[index, :, 5] = max_shape[1]  # width
            masks_batch[index, :, 6] = max_shape[0]  # height

            # add flattened mask
            for mask_index, mask in enumerate(annotations['masks']):
                masks_batch[index, mask_index, 7:7 + (mask.shape[0] * mask.shape[1])] = mask.flatten()

        return list(batches) + [masks_batch]
# _________________________________________________________________________________________________
"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from keras_maskrcnn.utils.overlap import compute_overlap
from keras_maskrcnn.utils.visualization import draw_masks

import numpy as np
import os
import time

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_masks      = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        # run network
        outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes  = outputs[-4]
        scores = outputs[-3]
        labels = outputs[-2]
        masks  = outputs[-1]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_masks      = masks[0, indices[scores_sort], :, :, image_labels]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            # draw_annotations(raw_image, generator.load_annotations(i)[0], label_to_name=generator.label_to_name)
            #draw_detections(raw_image, image_boxes, image_scores, image_labels, score_threshold=score_threshold, label_to_name=generator.label_to_name)
            draw_masks(raw_image, image_boxes.astype(int), image_masks, labels=image_labels)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
            all_masks[i][label]      = image_masks[image_detections[:, -1] == label, ...]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections, all_masks


def _get_annotations(generator):
    """ Get the ground truth anno from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = anno[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth anno.
    # Returns
        A list of lists containing the anno for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_masks       = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the anno
        anno = generator.load_annotations(i)
        anno['masks'] = np.stack(anno['masks'], axis=0)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = anno['bboxes'][anno['image_labels'] == label, :].copy()
            all_masks[i][label]       = anno['masks'][anno['image_labels'] == label, ..., 0].copy()

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_annotations, all_masks


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    binarize_threshold=0.5,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator          : The generator that represents the dataset to evaluate.
        model              : The model to evaluate.
        iou_threshold      : The threshold used to consider when a detection is positive or negative.
        score_threshold    : The score confidence threshold to use for detections.
        max_detections     : The maximum number of detections to use per image.
        binarize_threshold : Threshold to binarize the masks with.
        save_path          : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and anno
    all_detections, all_masks     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations, all_gt_masks = _get_annotations(generator)
    average_precisions = {}

    # import pickle
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_masks, open('all_masks.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))
    # pickle.dump(all_gt_masks, open('all_gt_masks.pkl', 'wb'))

    # process detections and anno
    for label in range(generator.num_classes()):
        false_positives = []
        true_positives  = []
        scores          = []
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            masks                = all_masks[i][label]
            anno          = all_annotations[i][label]
            gt_masks             = all_gt_masks[i][label]
            num_annotations     += anno.shape[0]
            detected_annotations = []

            for d, mask in zip(detections, masks):
                box = d[:4].astype(int)
                scores.append(d[4])

                if anno.shape[0] == 0:
                    false_positives.append(1)
                    true_positives.append(0)
                    continue

                if box[3] > gt_masks[0].shape[0]:
                    print('Box 3 error: {} Fix {} -> {}'.format(box, box[3], gt_masks[0].shape[0]))
                    box[3] = gt_masks[0].shape[0]
                if box[2] > gt_masks[0].shape[1]:
                    print('Box 2 error: {} Fix {} -> {}'.format(box, box[2], gt_masks[0].shape[1]))
                    box[2] = gt_masks[0].shape[1]
                if box[0] < 0:
                    print('Box 0 error: {} Fix {} -> {}'.format(box, box[0], 0))
                    box[0] = 0
                if box[1] < 0:
                    print('Box 1 error: {} Fix {} -> {}'.format(box, box[1], 0))
                    box[1] = 0

                # resize to fit the box
                mask = cv2.resize(mask, (box[2] - box[0], box[3] - box[1]))

                # binarize the mask
                mask = (mask > binarize_threshold).astype(np.uint8)

                # place mask in image frame
                mask_image = np.zeros_like(gt_masks[0])
                mask_image[box[1]:box[3], box[0]:box[2]] = mask
                mask = mask_image

                overlaps            = compute_overlap(np.expand_dims(mask, axis=0), gt_masks)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives.append(0)
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives.append(1)
                    true_positives.append(0)

        # no anno -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        false_positives = np.array(false_positives, dtype=np.uint8)
        true_positives = np.array(true_positives, dtype=np.uint8)
        scores = np.array(scores)

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


class Evaluate(keras.callbacks.Callback):
    def __init__(
        self,
        generator,
        iou_threshold=0.5,
        score_threshold=0.01,
        max_detections=300,
        save_map_path=None,
        binarize_threshold=0.5,
        tensorboard=None,
        weighted_average=False,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator          : The generator that represents the dataset to evaluate.
            iou_threshold      : The threshold used to consider when a detection is positive or negative.
            score_threshold    : The score confidence threshold to use for detections.
            max_detections     : The maximum number of detections to use per image.
            binarize_threshold : The threshold used for binarizing the masks.
            save_path          : The path to save images with visualized detections to.
            tensorboard        : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average   : Compute the mAP using the weighted average of precisions among mid_to_no.
            verbose            : Set the verbosity level, by default this is set to 1.
        """
        self.generator        = generator
        self.iou_threshold    = iou_threshold
        self.score_threshold  = score_threshold
        self.max_detections   = max_detections
        self.save_map_path    = save_map_path
        self.tensorboard      = tensorboard
        self.weighted_average = weighted_average
        self.verbose          = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        # run evaluation
        start_time = time.time()
        average_precisions = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=None
        )

        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = mean_ap
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        if self.verbose == 1:
            print('Time: {:.2f} mAP: {:.4f}'.format(time.time() - start_time, mean_ap))

        if self.save_map_path is not None:
            out = open(self.save_map_path, 'a')
            out.write('Ep {}: mAP: {:.4f}\n'.format(epoch + 1, mean_ap))
            out.close()

        logs['mAP'] = mean_ap

# _________________________________________________________________________________________________


#
# import base64
# import zlib
#
# import numpy as np
# import pandas as pd
# from cv2 import cv2
# from pycocotools import mask as coco_mask
#
#
# def encode_binary_mask(mask):
#     """Converts a binary mask into OID challenge encoding ascii text."""
#
#     # convert input mask to expected COCO API input --
#     mask_to_encode = np.expand_dims(mask, axis=2)
#     mask_to_encode = np.asfortranarray(mask_to_encode)
#
#     # RLE encode mask --
#     encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]
#
#     # compress and base64 encoding --
#     binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
#     # binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_SPEED)
#     base64_str = base64.b64encode(binary_str)
#     return base64_str
#
#
# def decode_binary_mask(mask, width, height):
#     """Converts a binary mask into OID challenge encoding ascii text."""
#
#     compressed_mask = base64.b64decode(mask)
#     rle_encoded_mask = zlib.decompress(compressed_mask)
#     # print(rle_encoded_mask)
#     decoding_dict = {
#         'size': [height, width],  # [im_height, im_width],
#         'counts': rle_encoded_mask
#     }
#     mask_tensor = coco_mask.decode(decoding_dict)
#     return mask_tensor
#
#
# def show_image(im, name='image'):
#     cv2.imshow(name, im.astype(np.uint8))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# def read_single_image(path):
#     img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
#     return img
#
#
# def get_class_arr(path, type='name'):
#     s = pd.read_csv(path, names=['google_name', 'name'], header=None)[type].values
#     return s
#
#
# def show_image_debug(draw, boxes, scores, labels, masks, classes):
#     from keras_retinanet.utils.visualization import draw_box, draw_caption
#     from keras_maskrcnn.utils.visualization import draw_mask
#     from keras_retinanet.utils.colors import label_color
#
#     # visualize detections
#     limit_conf = 0.2
#     for box, score, label, mask in zip(boxes, scores, labels, masks):
#         # scores are sorted so we can break
#         if score < limit_conf:
#             break
#
#         color = label_color(label)
#         color_mask = (255, 0, 0)
#
#         b = box.astype(int)
#         draw_box(draw, b, color=color)
#
#         mask = mask[:, :]
#         draw_mask(draw, b, mask, color=color_mask)
#
#         caption = "{} {:.3f}".format(classes[label], score)
#         print(caption)
#         draw_caption(draw, b, caption)
#     draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
#     show_image(draw)
#     # cv2.imwrite('debug.png', draw)
#
#
# def get_preds_as_string(id, input_image, boxes, scores, labels, masks, classes_google):
#     thr_keep_in_predictions = 0.01
#     thr_mask = 0.5
#     shape0, shape1 = input_image.shape[0], input_image.shape[1]
#     s1 = '{},{},{},'.format(id, shape1, shape0)
#
#     for i in range(scores.shape[0]):
#         score = scores[i]
#
#         if score < thr_keep_in_predictions:
#             continue
#
#         box = boxes[i]
#         label = classes_google[labels[i]]
#         mask = masks[i]
#
#         x1 = int(box[0] * shape1)
#         y1 = int(box[1] * shape0)
#         x2 = int(box[2] * shape1)
#         y2 = int(box[3] * shape0)
#         mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
#
#         mask[mask > thr_mask] = 1
#         mask[mask <= thr_mask] = 0
#         mask_complete = np.zeros((shape0, shape1), dtype=np.uint8)
#         mask_complete[y1:y2, x1:x2] = mask
#
#         enc_mask = encode_binary_mask(mask_complete)
#         str1 = str(label) + ' ' + str(score) + ' '
#         str1 += str(enc_mask)[2:-1] + ' '
#         s1 += '{} {:.8f} {} '.format(label, score, str(enc_mask)[2:-1])
#
#     s1 += '\n'
#     return s1


class Segmentation:
    # MaskPath,ImageID,LabelName,BoxID,BoxXMin,BoxXMax,BoxYMin,BoxYMax,PredictedIoU,Clicks
    def __init__(self,
                 image_id,
                 mask_path,
                 label_name,
                 box_id,
                 box_x_min,
                 box_x_max,
                 box_y_min,
                 box_y_max,
                 predicted_iou=None,
                 clicks=None,
                 ):
        self.image_id      = image_id
        self.mask_path     = mask_path
        self.label_name    = label_name
        self.box_id        = box_id
        self.box_x_min     = box_x_min
        self.box_x_max     = box_x_max
        self.box_y_min     = box_y_min
        self.box_y_max     = box_y_max
        self.predicted_iou = predicted_iou
        self.clicks        = clicks

    def __repr__(self):
        return f'label_name: {self.label_name}, box: {self.box_x_min, self.box_x_max, self.box_y_min, self.box_y_max}'

class SampleAnnoDesc:
    def __init__(self,
                 image_id,
                 boxes         = None,  # Annotation of boxes;
                 segmentations = None,  # Annotation of segmentations;
                 relationships = None,  # Annotation of relationships;
                 narratives    = None,  # Annotation of local narratives;
                 image_labels  = None,  # Annotation of image level image_labels;
                 ):
        self.image_id      = image_id
        self.boxes         = boxes
        self.segmentations = segmentations
        self.relationships = relationships
        self.narratives    = narratives
        self.image_labels  = image_labels

    def __repr__(self):
        return f'image_id: {self.image_id}, segmentations: {self.segmentations}'

def get_class_names(class_names_csv):
    """return MID vs. class_name dict."""
    names = pd.read_csv(class_names_csv, names=['MID', 'class_name'], index_col='MID')
    return names['class_name'].to_dict()


def get_sample_anno_desc_list(image_pattern     = None,
                              boxes_csv         = None,
                              segmentations_csv = None,
                              relationships_csv = None,
                              narratives_csv    = None,
                              image_labels_csv  = None, ):
    sample_anno_desc_list = []
    # image_id_anno_dict has structure:
    # {image_id: {'boxes'         : list of Box,
    #             'segmentations' : list of Segmentation,
    #             'relationships' : list of Relationship,
    #             'narratives'    : list of Narrative,
    #             'image_labels'  : list of ImageLabel}}
    image_id_anno_dict = {}

    def _ensure_image_id_anno_in_dict(image_id, anno=None):
        if image_id not in image_id_anno_dict.keys():
            image_id_anno_dict[image_id] = {}
        if anno is not None and anno not in image_id_anno_dict[image_id].keys():
            image_id_anno_dict[image_id][anno] = []

    if image_pattern is not None:
        from glob import glob
        image_filenames = glob(image_pattern)
        image_ids       = list(map(lambda s: os.path.splitext(os.path.basename(s))[0], image_filenames))
        for id in image_ids:
            _ensure_image_id_anno_in_dict(id)

    if boxes_csv         is not None: ...
    if segmentations_csv is not None:
        seg_df: pd.DataFrame = pd.read_csv(segmentations_csv, index_col='ImageID')
        groupby_df = seg_df.groupby('ImageID')
        for image_id, gdf in groupby_df:
            _ensure_image_id_anno_in_dict(image_id, 'segmentations')
            for _, row in gdf.iterrows():
                image_id_anno_dict[image_id]['segmentations'].append(
                    Segmentation(
                        image_id      = image_id,
                        mask_path     = row['MaskPath'],
                        label_name    = row['LabelName'],
                        box_id        = row['BoxID'],
                        box_x_min     = row['BoxXMin'],
                        box_x_max     = row['BoxXMax'],
                        box_y_min     = row['BoxYMin'],
                        box_y_max     = row['BoxYMax'],
                        predicted_iou = row['PredictedIoU'],
                        clicks        = row['Clicks'],
                    )
                )
    if relationships_csv is not None: ...
    if narratives_csv    is not None: ...
    if image_labels_csv  is not None: ...

    for id, anno in image_id_anno_dict.items():
        sample_anno_desc_list.append(
            SampleAnnoDesc(
                id,
                boxes         = anno['boxes']         if 'boxes'         in anno.keys() else None,
                segmentations = anno['segmentations'] if 'segmentations' in anno.keys() else None,
                relationships = anno['relationships'] if 'relationships' in anno.keys() else None,
                narratives    = anno['narratives']    if 'narratives'    in anno.keys() else None,
                image_labels  = anno['image_labels']  if 'image_labels'  in anno.keys() else None,
            )
        )

    return sample_anno_desc_list


if __name__ == '__main__':
    import os
    ANNO_DIR = 'D:/venv-tensorflow2/ins-5th/open-images-dataset/annotation-instance-segmentation/'
    CLASS_NAMES_CSV   = os.path.join(ANNO_DIR, 'metadata/challenge-2019-mid_to_no-description-segmentable.csv')
    SEGMENTATIONS_CSV = os.path.join(ANNO_DIR, 'train/challenge-2019-train-masks/challenge-2019-train-segmentation-masks.csv')

    descs = get_sample_anno_desc_list(
        segmentations_csv=SEGMENTATIONS_CSV,
    )
    # descs = get_sample_anno_desc_list(image_pattern='./sample_images/*.jpg')


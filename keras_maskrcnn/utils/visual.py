#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

import os

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

from cv2 import cv2
from matplotlib import pyplot as plt

from utils.ins_utils import CSVGenerator
# from bin2.tf_dataset_utils import *

# Configurations: ______________________________________________________________________________________________________
RUN_ON = 'local' if os.path.exists('C:/') else \
         'kaggle' if os.path.exists('/kaggle') else \
         'gcp'

DATASET_DIR = 'D:/venv-tensorflow2/open-images-dataset' if RUN_ON == 'local' else \
              'gs://tyu-ins-sample'

TEST_IMAGE_DIR = f'{DATASET_DIR}/test'
MASK_CSV_PATH  = f'{DATASET_DIR}/annotation-instance-segmentation/train/' \
                 f'challenge-2019-train-masks/challenge-2019-train-segmentation-masks.csv'
KLASS_CSV_PATH = f'{DATASET_DIR}/annotation-instance-segmentation/' \
                 f'metadata/challenge-2019-classes-description-segmentable.csv'

# class_names.columns: ['MID', 'class_name'], where 'MID' is 'label_name'
klass_df: pd.DataFrame = pd.read_csv(KLASS_CSV_PATH, names=['MID', 'class_name'])
N_CLASS = len(klass_df)

if 'color for class':
    colors = list(map(lambda h: (matplotlib.colors.hsv_to_rgb([h, 1., 1.]) * 255).astype('uint8'),
                      np.arange(0, 1, 1.0 / N_CLASS)))
    np.random.seed(2020)
    np.random.shuffle(colors)
    klass_df['color'] = colors

LABEL_TO_MID = klass_df['MID'].to_dict()
MID_TO_NAME  = klass_df.set_index('MID')['class_name'].to_dict()
MID_TO_COLOR = klass_df.set_index('MID')['color'].to_dict()

BOX_COLOR    = (  0, 255,  64)
BORDER_COLOR = (255, 255, 128)
NAME_COLOR   = (255, 255, 255)


def dirname_ntimes(path_or_dir, ntimes=1):
    ret = path_or_dir
    for _ in range(ntimes):
        ret = os.path.dirname(ret)
    return ret


def id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def array_map(fn, *arrays, map_start_axis):
    """for elem in arrays, elem.shape[: map_start_axis] should identical."""
    fix_shape = arrays[0].shape[: map_start_axis]

    arrays = [a.reshape([-1, *a.shape[map_start_axis:]]) for a in arrays]
    arrays = np.stack(list(map(fn, *arrays)), axis=0)

    return arrays.reshape([*fix_shape, *arrays.shape[1:]])


def show(arr):
    plt.imshow(arr)
    plt.show()


class Anno:
    def __init__(self, mask_id, mid, box_norm, example=None,
                 box_id=None, predicted_iou=None, clicks=None):
        # mask_path actually is mask filename.
        self.mask_id       = self.anno_id = mask_id
        self.mid           = mid
        self.name          = MID_TO_NAME[self.mid]
        self.color         = MID_TO_COLOR[self.mid]
        self.box_id        = box_id
        self.box_norm      = np.array(box_norm)
        self.predicted_iou = predicted_iou
        self.clicks        = clicks
        self.example       = example
        # origin mask, shape: [h, w], dtype: uint8, value: [0, 255]. use get_mask to access this field!
        self._mask         = None
        self.h             = None
        self.w             = None

    def get_mask(self):
        return self._mask if self._mask is not None else \
               self.example.group.load_mask_fn(self.mask_id)

    def set_mask(self, mask):
        """mask: shape: [h, w], dtype: uint8, value: [0, 255]"""
        self._mask     = mask
        self.h, self.w = mask.shape[:2]
        return self

    def box_to_show(self, h=None, w=None, color=BOX_COLOR, thickness=2):
        """:return array, shape: [w, h, 3], dtype: uint8, value: [0, 255]"""
        if h is None or w is None: h, w = self.h, self.w
        x1, y1, x2, y2 = (self.box_norm * [w, h, w, h]).astype(np.int)
        return cv2.rectangle(np.zeros(shape=(h, w, 3), dtype=np.uint8),
                             pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness)

    def name_to_show(self, h=None, w=None, color=NAME_COLOR, thickness=2):
        """:return array, shape: [w, h, 3], dtype: uint8, value: [0, 255]"""
        if h is None or w is None: h, w = self.h, self.w
        x1, y1, x2, y2 = (self.box_norm * [w, h, w, h]).astype(np.int)
        return cv2.putText(np.zeros(shape=(h, w, 3), dtype=np.uint8), self.name,
                           org=(x1 + 10, y1 + 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=1., color=color, thickness=thickness)

    def mask_to_show(self, new_wh=None, use_color=True):
        """:return array, shape: [w, h, 3], dtype: uint8, value: [0, 255]"""
        mask  = self.get_mask().copy()
        # resize to adapt other image.
        mask  = cv2.resize(mask, dsize=new_wh) if new_wh else mask
        # [0, 255] to [0, 1]
        mask  = (mask > 127).astype('uint8')
        # use color or black-white.
        color = self.color if use_color else np.array([255, 255, 255])

        return mask[..., np.newaxis] * color[np.newaxis, np.newaxis, :]

    def border_to_show(self, new_wh=None, color=BORDER_COLOR, thickness=2):
        """:return array, shape: [w, h, 3], dtype: uint8, value: [0, 255]"""
        mask  = self.get_mask().copy()
        # resize to adapt other image.
        mask  = cv2.resize(mask, dsize=new_wh) if new_wh else mask
        # shape: [h, w], dtype: uint8
        border = cv2.dilate(cv2.Canny(mask, 10, 30),
                            kernel=np.ones([thickness, thickness], np.uint8), iterations=3)
        # [0, 255] to [0, 1]
        border = (border > 127).astype('uint8')

        return border[..., np.newaxis] * np.array(color, dtype=np.uint8)[np.newaxis, np.newaxis, :]

    def __repr__(self):
        return f'    mask_id: {self.mask_id:38s}, box_norm: {self.box_norm}'


class Example:
    def __init__(self, image_id, group=None):
        self.image_id  = self.example_id = image_id
        self.group     = group
        # origin image, shape: [h, w, 3], dtype: uint8, use get_image to access this field!
        self._image    = None
        self.annos     = {}

    def add_anno(self, anno: Anno):
        self.annos[anno.mask_id] = anno
        anno.example = self
        return self

    def get_anno(self, anno_id): return self.annos[anno_id]

    def get_image(self):
        return self._image if self._image is not None else \
               self.group.load_image_fn(self.image_id)

    def set_image(self, image):
        self._image = image
        return self

    def get_annotated_image(self, draw_masks=True, draw_border=True, draw_boxes=True, draw_name=True):
        """based on image's h/w ."""
        image = self.get_image().copy()
        h, w  = image.shape[:2]
        for _, anno in self.annos.items():
            if draw_masks:
                mask   = anno.mask_to_show(new_wh=(w, h))
                image  = np.where(np.max(mask, axis=-1, keepdims=True) > 127,
                                 0.5 * image + 0.5 * mask, image).astype(np.uint8)
            if draw_border:
                border = anno.border_to_show(new_wh=(w, h))
                image  = np.where(np.max(border, axis=-1, keepdims=True) > 127, border, image)
            if draw_boxes:
                box    = anno.box_to_show(h, w)
                image  = np.where(np.max(box, axis=-1, keepdims=True) > 127, box, image)
            if draw_name:
                name   = anno.name_to_show(h, w)
                image  = np.where(np.max(name, axis=-1, keepdims=True) > 127, name, image)

        return image

    def __repr__(self):
        ret = f'    image_id: {self.image_id}, annotations: \n'
        for _, a in self.annos.items():
            ret += f'    {a}\n'
        return ret


class ExampleGroup:
    def __init__(self, src_args: dict=None):
        self.examples     = {}
        self.load_image_fn = None
        self.load_mask_fn  = None
        self.src_args     = src_args

    def add_example(self, exam):
        self.examples[exam.image_id] = exam
        exam.group = self
        return self

    def get_example(self, exam_id): return self.examples[exam_id]

    def get_image(self, image_id) : return self.get_example(image_id).get_image()
                                    
    def get_anno(self, anno_id)   : return self.examples[anno_id.split('_')[0]].get_anno(anno_id)
                                    
    def get_mask(self, mask_id)   : return self.get_anno(mask_id).get_mask()

    def show_example(self, exam_id, draw_masks=True, draw_borders=True, draw_boxes=True, draw_names=True):
        exam = self.get_example(exam_id)
        arr  = exam.get_annotated_image(draw_masks, draw_borders, draw_boxes, draw_names)
        show(arr)

    def show_separate(self, exam_id):
        exam  = self.get_example(exam_id)
        image = exam.get_image()

        show(image)
        for _, a in exam.annos.items():
            show(a.mask_to_show() + a.border_to_show())
            show(a.box_to_show() + a.name_to_show())

    def show_many(self, exam_ids):
        for id in exam_ids:
            self.show_example(id)

    def show_all(self):
        for id in self.examples.keys():
            self.show_example(id)

    def __repr__(self):
        ret = 'examples ' + '='*100 + '\n'
        for _, e in self.examples.items():
            ret += f'{e}\n'
        return ret

    @staticmethod
    def from_mask_csv(mask_csv_path, image_ids=None, images_dir=None, masks_dir=None, tvt=None):
        if tvt is None:
            tvt = 'train' if 'train' in mask_csv_path else 'validation'
        if images_dir is None:
            images_dir = os.path.join(dirname_ntimes(mask_csv_path, 4), tvt)
        if masks_dir is None:
            masks_dir = os.path.join(dirname_ntimes(mask_csv_path, 2), 'all-masks')

        def load_image(image_id):
            path = os.path.join(images_dir, image_id + '.jpg')
            return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        def load_mask(mask_id):
            path = os.path.join(masks_dir, mask_id + '.png')
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        group = ExampleGroup(src_args={'images_dir': images_dir,
                                       'masks_dir' : masks_dir,
                                       'tvt'       : tvt, })
        group.load_image_fn = load_image
        group.load_mask_fn  = load_mask
        # columns: ['ImageID', 'MaskPath', 'LabelName', 'BoxID', 'BoxXMin', 'BoxXMax',
        #           'BoxYMin', 'BoxYMax', 'PredictedIoU', 'Clicks']
        mask_df = pd.read_csv(mask_csv_path)
        if image_ids is not None:
            mask_df = mask_df[mask_df['ImageID'].isin(image_ids)]
        for _, row in mask_df.iterrows():
            # 'MaskPath' here actually is mask file name(e.g. 88e582a7b14e34a8_m039xj__6133896f.png)
            mask_fname, image_id, mid, x1, x2, y1, y2 = \
                row[['MaskPath', 'ImageID', 'LabelName', 'BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax']]
            mask_id = os.path.splitext(mask_fname)[0]

            if image_id not in group.examples:
                group.add_example(Example(image_id))

            group.examples[image_id].add_anno(Anno(mask_id, mid, np.array([x1, y1, x2, y2])))

        return group

    @staticmethod
    def from_generator(gen: CSVGenerator, i_batch):
        bat_image_preprocessed, (bat_reg, bat_cls, bat_mix) = gen[i_batch]
        # -1 axis struct: x1 + y1 + x2 + y2 + label + width + height + max_image_size
        bat_boxes_abs, bat_labels, bat_w, bat_h, bat_masks_flat = np.split(bat_mix, [4, 5, 6, 7], axis=-1)
        # shape: [B, N, ...]
        bat_image      = ((bat_image_preprocessed + 1.0) * 127).astype('uint8')
        bat_w          = bat_w.squeeze(axis=-1).astype('int')
        bat_h          = bat_h.squeeze(axis=-1).astype('int')
        bat_boxes_norm = bat_boxes_abs / np.stack([bat_w, bat_h, bat_w, bat_h], axis=-1)
        bat_masks      = array_map(lambda w, h, flat_mask: 255 * flat_mask.reshape([h, w]).astype(np.uint8),
                                   bat_w, bat_h, bat_masks_flat,
                                   map_start_axis=2)
        group          = gen.groups[i_batch]
        bat_image_id   = list(map(lambda i: id_from_path(gen.image_path(i)), group))
        bat_anno_descs = list(map(lambda i: gen.anno_descs[gen.image_path(i)], group))

        group = ExampleGroup()
        for image_id, image, annos, masks, boxes_norm in \
                zip(bat_image_id, bat_image, bat_anno_descs, bat_masks, bat_boxes_norm):
            example = Example(image_id).set_image(image)
            # annos has variant length.
            for i, a in enumerate(annos):
                anno = Anno(mask_id=id_from_path(a['mask_path']),
                            mid=a['label_name'],
                            box_norm=boxes_norm[i]
                            ).set_mask(masks[i])
                example.add_anno(anno)

            group.add_example(example)

        return group

    @staticmethod
    def from_tf_dataset(ds_record: tf.data.Dataset, image_ids=None, klass_csv_path=KLASS_CSV_PATH, batch_size=8):
        if 'process-tf-dataset':
            from keras_maskrcnn.utils.tf_dataset import parse_single_sequence_example, make_fn_decoder

            ds = ds_record.map(parse_single_sequence_example)
            if image_ids is not None:
                ds = ds.filter(lambda ctx, _: tf.reduce_any(tf.equal(ctx['image_id'], image_ids)))
            ds = ds.map(make_fn_decoder(klass_csv_path)).padded_batch(batch_size)

        def _bytes_to_str(x):
            return x.decode('utf-8') if not isinstance(x, np.ndarray) else \
                array_map(lambda a: a.decode('utf-8'), x, map_start_axis=x.ndim)

        if 'assemble batches to groups':
            groups = []
            for bat_tf_exam in ds: # assemble ExampleGroup
                # bat_tf_exam is a dict with structure of:
                # {'image_id': Tensor of shape: [B, ],
                #  'image'   : Tensor of shape: [B, H, W, C],
                #  'mask_ids': Tensor of shape: [B, M,],
                #  'boxes'   : Tensor of shape: [B, M, 4],
                #  'labels'  : Tensor of shape: [B, M,],
                #  'masks'   : Tensor of shape: [B, M, H, W, 1], ...}
                # where: B=batch_size, M=max_annotations_within_batch, H=batch_height, W=batch_width, C=batch_channels
                bat_image_id  = _bytes_to_str(bat_tf_exam['image_id'].numpy())
                bat_image     = bat_tf_exam['image'].numpy().astype(np.uint8)
                bat_mask_ids  = _bytes_to_str(bat_tf_exam['mask_ids'].numpy())
                bat_boxes_abs = bat_tf_exam['boxes'].numpy()
                bat_labels    = bat_tf_exam['labels'].numpy()
                bat_masks     = bat_tf_exam['masks'].numpy().squeeze(axis=-1).astype(np.uint8)
                B, H, W, C    = batch_size, batch_height, batch_width, batch_channels = bat_image.shape

                group = ExampleGroup()
                # for every example, assemble Example.
                for image_id, image, mask_ids, boxes_abs, labels, masks in \
                        zip(bat_image_id, bat_image, bat_mask_ids, bat_boxes_abs, bat_labels, bat_masks):
                    exam = Example(image_id).set_image(image)
                    # for every annotation, assemble Anno, each has length of M=max_annotations_within_batch
                    for mask_id, box_abs, label, mask in zip(mask_ids, boxes_abs, labels, masks):
                        if mask_id == '':
                            continue
                        anno = Anno(mask_id,
                                    mid=LABEL_TO_MID[label],
                                    box_norm=box_abs / np.array([W, H, W, H])
                                    ).set_mask(mask)
                        # add anno to example.
                        exam.add_anno(anno)
                    # add example to group.
                    group.add_example(exam)
                # append group to groups.
                groups.append(group)

        return groups

    @staticmethod
    def from_prediction(model_input, model_output):
        """
        model_input shape: [B, bat_h, bat_w, C]
        MaskRCNN Output: list(len=7) >> [
            'regression'    : ndarray(shape=[B, N, 4]),
            'classification': ndarray(shape=[B, N, n_class]),
            # 'others'      : [],
            'boxes_masks'   : ndarray(shape=[B, K, 235204]),
            'boxes'         : ndarray(shape=[B, K, 4]),
            'scores'        : ndarray(shape=[B, K]),
            'labels'        : ndarray(shape=[B, K]),
            'masks'         : ndarray(shape=[B, K, 28, 28, n_class]),
            # 'others'      : [],
        ]
        where: B is batch size, N is total anchors, K is top-K anchors
        :param model_output:
        :return:
        """
        B, H, W, C = batch_size, batch_height, batch_width, batch_channels = model_input.shape
        group = ExampleGroup()
        # for every example in batch.
        for i_exam, (inp, _, _, _, boxes, scores, labels, masks_n_class) in enumerate(zip(model_input, *model_output)):
            exam = Example(f'{i_exam}').set_image(
                cv2.cvtColor(inp.copy(), cv2.COLOR_BGR2RGB))
            # for every annotation in example, each has length of K=max_detections.
            for i_anno, (box_abs, score, label, mask_n_class) in enumerate(zip(boxes, scores, labels, masks_n_class)):
                if score < 0.5:
                    continue
                if 'create-Anno-instance':
                    # create Anno instance.
                    mask_id = f'{i_exam}_{i_anno}'
                    mid = LABEL_TO_MID[label]
                    box_norm = box_abs / np.array([W, H, W, H])
                    anno = Anno(mask_id, mid, box_norm, example=exam)
                if 'set mask':
                    x1, y1, x2, y2 = box_abs.astype(np.int)
                    h, w           = y2 - y1, x2 - x1
                    # mask_rel is relative mask within box, fill it in absolute mask.
                    mask_rel = mask_n_class[..., label].astype(np.float)
                    mask_rel = cv2.resize(mask_rel, dsize=(w, h))
                    mask_rel = (mask_rel > 0.5).astype(np.uint8) * 255

                    mask                 = np.zeros([H, W], dtype=np.uint8)
                    mask[y1: y2, x1: x2] = mask_rel

                    anno.set_mask(mask)
                # add anno to example.
                exam.add_anno(anno)
            # add example to group.
            group.add_example(exam)

        return group


# if __name__ == '__main__':
#
#     mask_df: pd.DataFrame = pd.read_csv(MASK_CSV_PATH)
#
#     if 'from-generator':
#         from albumentations import *
#         bbox_params = BboxParams(format='pascal_voc', min_area=1.0, min_visibility=0.1, label_fields=['labels'])
#
#         transform_generator = Compose([
#             HorizontalFlip(p=0.5),
#             OneOf([
#                 IAAAdditiveGaussianNoise(),
#                 GaussNoise(),
#             ], p=0.1),
#             OneOf([
#                 MotionBlur(p=.1),
#                 MedianBlur(blur_limit=3, p=.1),
#                 Blur(blur_limit=3, p=.1),
#             ], p=0.1),
#             ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=20, p=1,
#                              border_mode=cv2.BORDER_CONSTANT),
#             OneOf([
#                 # CLAHE(clip_limit=2),
#                 IAASharpen(),
#                 IAAEmboss(),
#                 RandomBrightnessContrast(),
#             ], p=0.1),
#             OneOf([
#                 RGBShift(p=1.0, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
#                 HueSaturationValue(p=1.0),
#             ], p=0.1),
#             ToGray(p=0.01),
#             ImageCompression(p=0.05, quality_lower=50, quality_upper=99),
#         ], bbox_params=bbox_params, p=1.0)
#
#         gen = CSVGenerator(
#             MASK_CSV_PATH,
#             KLASS_CSV_PATH,
#             image_dir=os.path.join(DATASET_DIR, 'train'),
#             transform_generator=transform_generator,
#             batch_size=4,
#             config=None,
#             image_min_side=800,
#             image_max_side=1024,
#             group_method=None,  # 'random',
#             is_rle=False,
#         )
#         group1 = ExampleGroup.from_generator(gen, 3)
#
#     if 'from-mask-csv':
#         group0 = ExampleGroup.from_mask_csv(MASK_CSV_PATH, image_ids=list(group1.examples.keys()))
#
#     if 'from-tf-dataset':
#         dir_tfrecord = '../ins-tfrecord'
#
#         image_ids = ['83fe588b6e01ef7a',
#                      '84c7775f391a85e4',
#                      '8736925341819488',
#                      '8746dc11f48a0d2c',
#                      '879c97c0e6abd298',
#                      '8836a53b50cb3e28',
#                      '884d87c19c1bdf88',
#                      '8952392367b3b61c']
#
#         fnames         = tf.io.matching_files(f'{dir_tfrecord}/train/*.tfrecord')
#         fnames         = tf.random.shuffle(fnames)
#         ds_fnames      = tf.data.Dataset.from_tensor_slices(fnames)  # .repeat()
#         ds_raw_example = ds_fnames.interleave(tf.data.TFRecordDataset,
#                                               cycle_length=len(fnames),
#                                               block_length=4,
#                                               num_parallel_calls=tf.data.experimental.AUTOTUNE,)
#         ds_raw_example = ds_raw_example.shuffle(buffer_size=100, reshuffle_each_iteration=False)
#         ds_raw_example = ds_raw_example.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
#         groups = ExampleGroup.from_tf_dataset(ds_raw_example, image_ids, batch_size=4)
#
#     if 'from-prediction':
#         from keras_maskrcnn.bin2.infer2 import *
#         if 'model' not in dir():
#             # model: tf.keras.Model = models.load_model(MODEL_PATH, backbone_name=BACKBONE)
#             backbone = models.backbone('resnet50')
#             model, *_ = create_models(
#                 backbone_retinanet=backbone.maskrcnn,
#                 num_classes=N_CLASS,
#                 weights='../ins/weights/mask_rcnn_resnet50_oid_v1.0.h5',
#                 class_specific_filter=False,
#                 anchor_params=None
#             )
#             model.run_eagerly = True
#         ds0 = tf.data.Dataset.from_tensor_slices(glob.glob(f'{TEST_IMAGE_DIR}/*.jpg'))
#         ds1 = ds0.map(read_jpg).padded_batch(BATCH_SIZE)
#         inp = list(ds1.take(1))[0]
#
#         # noinspection PyUnboundLocalVariable
#         pred = model.predict_on_batch(inp)
#
#         group3 = ExampleGroup.from_prediction(inp.numpy(), pred)


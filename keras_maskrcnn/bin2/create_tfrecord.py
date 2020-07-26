# coding: utf-8

# _________________________________________________________________________________________________
# Imports:
import argparse
import hashlib
import os
import sys
from math import ceil
from multiprocessing.pool import Pool

import pandas as pd
import tensorflow as tf
# _________________________________________________________________________________________________
# Configurations:
RUN_ON = 'local' if os.path.exists('C:/') else \
    'kaggle' if os.path.exists('/kaggle') else \
        'gcp'

if '--src' in sys.argv:
    # if cli is used, set config by cli args.
    parser = argparse.ArgumentParser(description='Convert open-images-dataset to tfrecord format.')
    parser.add_argument('--env', default='auto', choices=['local', 'kaggle', 'gcp', 'auto'])
    parser.add_argument('--src', required=True, help='Source Dataset.')
    parser.add_argument('--dst', required=True, help='Destination Dataset.')
    CONFIG = parser.parse_args().__dict__
    if CONFIG['env'] is 'auto':
        CONFIG['env'] = RUN_ON
else:
    # if don't use cli, use hard-coded config.
    if RUN_ON == 'local':
        dir_src_ds = 'D:/venv-tensorflow2/open-images-dataset'
        dir_dst_ds = f'D:/venv-tensorflow2/ins-tfrecord'
    else:
        dir_src_ds = 'gs://tyu-ins-sample'
        dir_dst_ds = 'gs://tyu-ins-sample-tfrecord'
    CONFIG = {
        'env': RUN_ON,
        'src': dir_src_ds,
        'dst': dir_dst_ds
    }
# post process of config or environment.
if CONFIG['env'] is 'local':
    os.chdir(r'D:\venv-tensorflow2\keras-maskrcnn')

N_SHARD_TRN = 100
N_SHARD_VAL = 10
N_SHARD_TST = 10

DIR_ANNO = f'{CONFIG["src"]}/annotation-instance-segmentation'
PATHS    = {
    'train': {
        'images'  : f'{CONFIG["src"]}/train',
        'mask_csv': f'{DIR_ANNO}/train/challenge-2019-train-masks/challenge-2019-train-segmentation-masks.csv',
        'masks'   : f'{DIR_ANNO}/train/all-masks'
    },
    'validation': {
        'images'  : f'{CONFIG["src"]}/validation',
        'mask_csv': f'{DIR_ANNO}/validation/challenge-2019-validation-masks/challenge-2019-validation-segmentation-masks.csv',
        'masks'   : f'{DIR_ANNO}/validation/all-masks'
    },
    'test': {
        'images'  : f'{CONFIG["src"]}/test'
    }
}
# _________________________________________________________________________________________________
# Functions :
def _bytes_list_feature(value):
    """value should be list of bytes. Returns BytesList Feature."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
    """value should be list of float. Returns FloatList Feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_list_feature(value):
    """value should be list of int64. Returns Int64List Feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value): return _bytes_list_feature([value])


def _float_feature(value): return _float_list_feature([value])


def _int64_feature(value): return _int64_list_feature([value])


def get_file_raw(dir, filename, ext=None):
    p = f'{dir}/{filename}' + (f'.{ext}' if ext is not None else '')
    return tf.io.read_file(p).numpy()


def gdf_to_example_desc(image_id_gdf):
    image_id, gdf = image_id_gdf
    return {
        'image_id'     : image_id,
        'li_mask_id'   : list(map(lambda a: a.split('.')[0], gdf['MaskPath'])),
        'li_label_name': list(map(lambda a: a, gdf['LabelName'])),
        'li_box_id'    : list(map(lambda a: a, gdf['BoxID'])),
        'li_box'       : list(map(lambda a: [a[1]['BoxXMin'], a[1]['BoxYMin'], a[1]['BoxXMax'], a[1]['BoxYMax']],
                                  gdf[['BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax']].iterrows())),
        'li_pred_iou'  : list(map(lambda a: a, gdf['PredictedIoU'])),
        'li_clicks'    : list(map(lambda a: a, gdf['Clicks'])),
    }


def write_tfrecord_trn_val(li_desc, dir_image, dir_mask, path_out_file):
    writer = tf.io.TFRecordWriter(path_out_file)
    for desc in li_desc:
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'image_id' : _bytes_feature(desc['image_id'].encode('utf-8')),
                'image_raw': _bytes_feature(get_file_raw(dir_image, desc['image_id'], ext='jpg')),
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'li_mask_id'   : tf.train.FeatureList(
                    feature=list(map(lambda a: _bytes_feature(a.split('.')[0].encode('utf-8')), desc['li_mask_id']))),
                'li_mask_raw'  : tf.train.FeatureList(
                    feature=list(map(lambda a: _bytes_feature(get_file_raw(dir_mask, a, ext='png')), desc['li_mask_id']))),
                'li_label_name': tf.train.FeatureList(
                    feature=list(map(lambda a: _bytes_feature(a.encode('utf-8')), desc['li_label_name']))),
                'li_box_id'    : tf.train.FeatureList(
                    feature=list(map(lambda a: _bytes_feature(a.encode('utf-8')), desc['li_box_id']))),
                'li_box'       : tf.train.FeatureList(
                    feature=list(map(lambda a: _float_list_feature(a), desc['li_box']))),
                'li_pred_iou'  : tf.train.FeatureList(
                    feature=list(map(lambda a: _float_feature(a), desc['li_pred_iou']))),
                'li_clicks'    : tf.train.FeatureList(
                    feature=list(map(lambda a: _bytes_feature(a.encode('utf-8')), desc['li_clicks']))),
            }),
        )
        writer.write(example.SerializeToString())
    writer.close()


def write_tfrecord_tst(li_image_id, dir_image, path_out_file):
    writer = tf.io.TFRecordWriter(path_out_file)
    for image_id in li_image_id:
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'image_id' : _bytes_feature(image_id.encode('utf-8')),
                'image_raw': _bytes_feature(get_file_raw(dir_image, image_id, ext='jpg')),
            })
        )
        writer.write(example.SerializeToString())
    writer.close()


def process_trn_val(subdataset_name, src_paths, output_dir, n_shard):
    """subdataset_name: 'train' or 'validation'; src_paths: dict with keys 'images', 'masks' and 'mask_csv'; """
    # Make sure output file dir exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # columns: 'MaskPath', 'ImageID', 'LabelName', 'BoxID',
    #          'BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax', 'PredictedIoU', 'Clicks'
    mask_df: pd.DataFrame = pd.read_csv(src_paths['mask_csv']).fillna('NaN')
    li_example_desc = list(map(gdf_to_example_desc, mask_df.groupby('ImageID')))
    # Sharding li_example_desc:
    li_shard = [[] for _ in range(n_shard)]
    for desc in li_example_desc:
        i_shard = int(hashlib.md5(desc['image_id'].encode('utf-8')).hexdigest(), 16) % n_shard
        li_shard[i_shard].append(desc)
    # Write tfrecord shards using multi-processing.
    p = Pool()
    for i_shard, shard in enumerate(li_shard):
        path_out_file = f'{output_dir}/{subdataset_name}-{i_shard:05d}-of-{n_shard:05d}.tfrecord'
        p.apply_async(write_tfrecord_trn_val, args=(shard, src_paths['images'], src_paths['masks'], path_out_file))
        # write_tfrecord(shard, src_paths['images'], src_paths['masks'], path_out_file)
    print(f'Processing dataset of {subdataset_name}...')
    p.close()
    p.join()
    print(f'Process dataset of {subdataset_name} done.')


def process_tst(src_paths, output_dir, n_shard):
    # Make sure output file dir exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fnames           = tf.io.gfile.listdir(src_paths['images'])
    images_per_shard = ceil(len(fnames) / n_shard)
    li_image_id_all  = list(map(lambda a: a.split('.')[0], fnames))
    li_shard = [[] for _ in range(n_shard)]
    for i_shard in range(n_shard):
        li_shard[i_shard] += li_image_id_all[i_shard * images_per_shard : (i_shard + 1) * images_per_shard]
    # Write tfrecord shards using multi-processing.
    p = Pool()
    for i_shard, shard in enumerate(li_shard):
        path_out_file = f'{output_dir}/test-{i_shard:05d}-of-{n_shard:05d}.tfrecord'
        p.apply_async(write_tfrecord_tst, args=(shard, src_paths['images'], path_out_file))
        # write_tfrecord(shard, src_paths['images'], src_paths['masks'], path_out_file)
    print(f'Processing dataset of test ...')
    p.close()
    p.join()
    print(f'Process dataset of test done.')


if __name__ == '__main__':
    process_trn_val('train',      PATHS['train'],      f'{CONFIG["dst"]}/train',      N_SHARD_TRN)
    process_trn_val('validation', PATHS['validation'], f'{CONFIG["dst"]}/validation', N_SHARD_VAL)
    process_tst(PATHS['test'], f'{CONFIG["dst"]}/test', N_SHARD_TST)


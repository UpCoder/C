import tensorflow as tf
import numpy as np
from glob import glob
import os
from dataset import config


def _parse_function_V1(proto):

    # define your tfrecord again. Remember that you saved your image as a string.
    # keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
    #                     "label": tf.FixedLenFeature([], tf.int64)}
    #
    keys_to_features = {
        'image': tf.FixedLenFeature([512, 512, 3], dtype=tf.int64, default_value=np.zeros([512, 512, 3])),
        'mask': tf.FixedLenFeature([512, 512, 3], dtype=tf.int64),
        'height': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64),
        'channel': tf.FixedLenFeature((), tf.int64),
    }
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)

    print(parsed_features)
    # parsed_features['image'] = tf.reshape(parsed_features['image'], [224, 224, 3])
    print(parsed_features['image'])
    return parsed_features['image'], parsed_features['height'], parsed_features['width'], parsed_features['mask']


def _parse_function_V2(proto):

    # define your tfrecord again. Remember that you saved your image as a string.
    # keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
    #                     "label": tf.FixedLenFeature([], tf.int64)}
    #
    keys_to_features = {
        'image': tf.FixedLenFeature([512, 512, 3], dtype=tf.float32,
                                    default_value=np.zeros([512, 512, 3], dtype=np.float)),
        'mask': tf.FixedLenFeature([512, 512, 3], dtype=tf.int64),
        'height': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64),
        'channel': tf.FixedLenFeature((), tf.int64),
    }
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)

    print(parsed_features)
    # parsed_features['image'] = tf.reshape(parsed_features['image'], [224, 224, 3])
    print(parsed_features['image'])
    return parsed_features['image'], parsed_features['height'], parsed_features['width'], parsed_features['mask']


def load_dataset(dataset_dir='/mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/tfrecords/V1', binary_flag=True):
    '''

    :param dataset_dir:
    :param binary_flag: 是否返回二分类的label
    :return:
    '''
    tfrecord_paths = []
    # for i in range(8):
    #     tfrecord_paths.append(os.path.join(dataset_dir, '{}.tfrecords'.format(i)))
    # print(tfrecord_paths)
    tfrecord_paths = glob(os.path.join(dataset_dir, '*.tfrecords'))
    print(tfrecord_paths)
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    if dataset_dir.endswith('V1'):
        dataset = dataset.map(_parse_function_V1, num_parallel_calls=config.num_parallel_reader)
    elif dataset_dir.endswith('V2'):
        dataset = dataset.map(_parse_function_V2, num_parallel_calls=config.num_parallel_reader)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(config.shuffle_size, seed=2019)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(config.num_prefecth)
    iterator = dataset.make_one_shot_iterator()
    images, height, width, mask = iterator.get_next()
    images = tf.reshape(images, [-1, 512, 512, 3])
    if binary_flag:
        mask = tf.cast(tf.greater(mask, 0), tf.uint8)
    else:
        mask = tf.cast(mask, tf.uint8)
    mask = mask[:, :, :, 1]
    input_shape = [config.batch_size]
    input_shape.extend(list(config.input_shape))
    images.set_shape(input_shape)
    mask.set_shape(input_shape[:3])
    mask = tf.expand_dims(mask, axis=3)
    tf.summary.image('input/images', tf.cast(images, tf.uint8), max_outputs=3)
    tf.summary.image('input/masks', tf.cast(mask * 200, tf.uint8), max_outputs=3)
    images = tf.cast(images, tf.float32)
    return images, height, width, mask


if __name__ == '__main__':
    print('')
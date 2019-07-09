import tensorflow as tf
import os
from tensorflow.contrib import slim
import numpy as np


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class Example:
    def __init__(self):
        self.attribute_names = []
        self.attribute_values = []
        self.attribute_types = []

    def init_values(self, names, types, values):
        if len(names) != len(types) and len(names) != len(values):
            assert False
        self.attribute_names = names
        self.attribute_types = types
        self.attribute_values = values

    def get_examples(self):
        features = {}
        for attribute_name, attribute_type, attribute_value in zip(self.attribute_names,
                                                                   self.attribute_types, self.attribute_values):
            if attribute_type == 'int':
                features[attribute_name] = _int64_feature(attribute_value)
            elif attribute_type == 'np_array':
                if attribute_value.dtype == 'uint8':
                    features[attribute_name] = _int64_feature(attribute_value.flatten().tolist())
                elif attribute_value.dtype == 'float32':
                    features[attribute_name] = _float64_feature(attribute_value.flatten().tolist())
                else:
                    assert False
            elif attribute_type == 'str':
                features[attribute_name] = _bytes_feature(attribute_value)
            else:
                assert False
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example


def convert_from_numpys(examples, tf_record_path):
    '''
    :param examples: the objects of Example
    :param tf_record_path
    :return:
    '''
    with tf.python_io.TFRecordWriter(tf_record_path) as writer:
        for example in examples:
            writer.write(example.get_examples().SerializeToString())


def main_write():
    import numpy as np
    example = Example()
    example.init_values(['image', 'height', 'width', 'channel'], ['np_array', 'int', 'int', 'int'],
                        [np.random.randint(0, 255, [224, 224, 3]), 224, 224, 3])
    convert_from_numpys([example], './tmp/test.tfrecords')


def read_tfrecords(tf_record_paths):
    '''

    :param tf_record_path:
    :return:
    '''
    keys_to_features = {
        'image': tf.VarLenFeature(tf.float32),
        'mask': tf.VarLenFeature(tf.int64),
        'height': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64),
        'channel': tf.FixedLenFeature((), tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Tensor('image'),
        'mask': slim.tfexample_decoder.Tensor('mask'),
        'height': slim.tfexample_decoder.Tensor('height'),
        'width': slim.tfexample_decoder.Tensor('width'),
        'channel': slim.tfexample_decoder.Tensor('channel'),

    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    reader = tf.TFRecordReader
    items_to_descriptions = {
        'image': 'A color image of varying height and width.',
        'height': 'Shape of the image',
        'width': 'A list of bounding boxes, one per each object.',
    }

    return slim.dataset.Dataset(
        data_sources=tf_record_paths,
        reader=reader,
        decoder=decoder,
        num_samples=1,
        items_to_descriptions=items_to_descriptions,
        )


def _parse_function(proto):

    # define your tfrecord again. Remember that you saved your image as a string.
    # keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
    #                     "label": tf.FixedLenFeature([], tf.int64)}
    #
    keys_to_features = {
        'image': tf.FixedLenFeature([512, 512, 3], dtype=tf.float32, default_value=np.zeros([512, 512, 3])),
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
    return parsed_features['image'], parsed_features['height'], parsed_features['width']


def demo_read():
    from glob import glob
    tfrecord_paths = glob(os.path.join('/mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/tfrecords/V1/*.tfrecords'))
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    dataset = dataset.map(_parse_function, num_parallel_calls=5)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(10, seed=2019)
    dataset = dataset.batch(32)
    # dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    images, height, width = iterator.get_next()
    images = tf.reshape(images, [-1, 512, 512, 3])

    # images = tf.cast(images, tf.float32)
    return images, height, width


def main_reader():
    import cv2
    from glob import glob
    # tfrecord_paths = glob(os.path.join('/mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/tfrecords/V1/*.tfrecords'))
    # dataset = read_tfrecords(tf_record_paths=tfrecord_paths)
    # batch_size = 100
    # provider = slim.dataset_data_provider.DatasetDataProvider(
    #     dataset, num_readers=1, common_queue_capacity=batch_size * 1000, common_queue_min=batch_size * 100, shuffle=True)

    # i_array, mask_array, height_tensor, width_tensor = provider.get(['image', 'mask', 'height', 'width'])
    i_array, height_tensor, width_tensor = demo_read()
    print(i_array)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    thread = tf.train.start_queue_runners(sess=sess)

    ia = sess.run([i_array])
    ia = np.squeeze(ia)
    print(np.shape(ia))
    for idx, image in enumerate(ia):
        cv2.imwrite('./tmp/{}.jpg'.format(idx), image[:, :, ::-1])


if __name__ == '__main__':
    # main_write()
    main_reader()

# -*- coding=utf-8 -*-
import sys
sys.path.insert(0, '.')
from glob import glob
import os
from dataset.utils import load_itk
import numpy as np
import cv2
from dataset.tfrecords_utils import Example
import tensorflow as tf
from tqdm import tqdm


def generate_tfrecords_V1Core(case_ids, dataset_dir, annotation_dir, save_path):
    num_frames = 0
    WW = 1750.
    WC = -525.

    low_boundary = WC - WW / 2
    up_boundary = WC + WW / 2
    with tf.python_io.TFRecordWriter(save_path) as writer:
        for case_id in tqdm(case_ids):
            cur_image_path = os.path.join(dataset_dir, '{}.mhd'.format(case_id))
            image, _, _ = load_itk(cur_image_path)

            # 开始进行截断操作
            image = np.asarray(image, np.float)
            image[image < low_boundary] = low_boundary
            image[image > up_boundary] = up_boundary
            image = (image - low_boundary) / (WW * 1.0)
            image = np.asarray(image * 255., np.int)
            # 结束

            cur_mask_path = os.path.join(annotation_dir, '{}.mhd'.format(case_id))
            mask, _, _ = load_itk(cur_mask_path)
            sum_slices = np.sum(np.sum(mask, axis=1), axis=1)
            num_pos_slices = np.sum(sum_slices != 0)
            pos_slice_idx_set = set(np.where(sum_slices != 0)[0].tolist())
            neg_slice_idx_set = set(range(len(image))) - pos_slice_idx_set
            pos_slice_idx_set = list(pos_slice_idx_set)
            neg_slice_idx_set = list(neg_slice_idx_set)
            np.random.shuffle(neg_slice_idx_set)
            neg_slice_idx_set = neg_slice_idx_set[:num_pos_slices]
            save_image_slices = []
            save_mask_slices = []
            for idx in pos_slice_idx_set:
                last_idx = idx - 1
                if last_idx < 0:
                    last_idx = 0
                next_idx = idx + 1
                if next_idx >= len(image):
                    next_idx = len(image) - 1

                cur_slices = np.concatenate(
                    [
                        [image[last_idx, :, :]], [image[idx, :, :]], [image[next_idx, :, :]]
                    ], axis=0
                )
                cur_mask_slices = np.concatenate(
                    [
                        [mask[last_idx, :, :]], [mask[idx, :, :]], [mask[next_idx, :, :]]
                    ], axis=0
                )
                save_image_slices.append(np.transpose(cur_slices, axes=[1, 2, 0]))
                save_mask_slices.append(np.transpose(cur_mask_slices, axes=[1, 2, 0]))

            for idx in neg_slice_idx_set:
                last_idx = idx - 1
                if last_idx < 0:
                    last_idx = 0
                next_idx = idx + 1
                if next_idx >= len(image):
                    next_idx = len(image) - 1

                cur_slices = np.concatenate(
                    [
                        [image[last_idx, :, :]], [image[idx, :, :]], [image[next_idx, :, :]]
                    ], axis=0
                )
                cur_mask_slices = np.concatenate(
                    [
                        [mask[last_idx, :, :]], [mask[idx, :, :]], [mask[next_idx, :, :]]
                    ], axis=0
                )
                save_image_slices.append(np.transpose(np.asarray(cur_slices, np.float32), axes=[1, 2, 0]))
                save_mask_slices.append(np.transpose(cur_mask_slices, axes=[1, 2, 0]))
            print(np.shape(save_image_slices), np.shape(save_mask_slices))

            for idx, (image, mask) in enumerate(zip(save_image_slices, save_mask_slices)):
                example = Example()
                example.init_values(
                    ['image', 'mask', 'height', 'width', 'channel'],
                    ['np_array', 'np_array', 'int', 'int', 'int'],
                    [np.asarray(image, np.uint8), np.asarray(mask, np.uint8), 512, 512, 3]
                )
                writer.write(example.get_examples().SerializeToString())
            num_frames += len(save_image_slices)
            # 保存图像
            # cur_tmp_dir = os.path.join(tmp_dir, case_id)
            # if not os.path.exists(cur_tmp_dir):
            #     os.mkdir(cur_tmp_dir)
            # for idx, (image, mask) in enumerate(zip(save_image_slices, save_mask_slices)):
            #     image[image > up_boundary] = up_boundary
            #     image[image < low_boundary] = low_boundary
            #     image = (image - low_boundary) / WW
            #     cv2.imwrite('{}/mask_{}.jpg'.format(cur_tmp_dir, idx), np.asarray(mask != 0, np.uint8) * 200)
            #     cv2.imwrite('{}/{}.jpg'.format(cur_tmp_dir, idx), np.asarray(image*255, np.uint8))
    return num_frames


def generate_tfrecords_V1(dataset_dir, annotation_dir, save_dir):
    '''
    生成tfrecords, 截断，使用原始的像素值
    保存的对象包括所有的具有object 的slice和一部分negative的slice, 每个样本选择negative slice和positive slice一样
    :param dataset_dir:
    :param annotation_dir:
    :return:
    '''
    image_paths = glob(os.path.join(dataset_dir, '*.mhd'))
    print(len(image_paths))
    case_ids = [os.path.basename(image_path).split('.')[0] for image_path in image_paths]
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    save_dir = os.path.join(save_dir, 'V1')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    num_processing = 16
    import multiprocessing
    # names = names[:40]
    num_id_per_processing = int(len(case_ids) / num_processing + 1)
    p = multiprocessing.Pool(processes=num_processing)
    results = []
    for i in range(num_processing):
        input_ids = case_ids[num_id_per_processing * i: num_id_per_processing * (i + 1)]
        print(len(input_ids))
        results.append(p.apply_async(generate_tfrecords_V1Core, args=(input_ids, dataset_dir, annotation_dir,
                                                                      os.path.join(save_dir,
                                                                                   '{}.tfrecords'.format(i)),)))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    total_slice_num = 0.0
    for result in results:
        total_slice_num += result.get()
    print('All subprocesses done.')
    print('the total_slice num is ', total_slice_num)


if __name__ == '__main__':
    dataset_dir = '/mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/train'
    annotation_dir = '/mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/annotation'
    save_dir = '/mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/tfrecords'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    generate_tfrecords_V1(dataset_dir, annotation_dir, save_dir)
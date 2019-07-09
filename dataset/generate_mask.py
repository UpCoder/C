# -*- coding=utf-8 -*-
import sys
sys.path.insert(0, '.')
import os
from dataset.utils import resolve_annotation, load_itk, world2pixel, save_mhd_image
import numpy as np
from tqdm import tqdm
from glob import glob


def generate_masks(annotation_path, dataset_dir, save_dir):
    '''
    根据annotation生成mask
    :param annotation_path:
    :param dataset_dir:
    :return:
    '''
    annotation_obj = resolve_annotation(annotation_path)
    image_paths = glob(os.path.join(dataset_dir, '*.mhd'))
    case_ids = [os.path.basename(image_path).split('.')[0] for image_path in image_paths]
    for case_id in tqdm(case_ids):
        # if case_id != '641414':
        #     continue
        mhd_path = os.path.join(dataset_dir, '{}.mhd'.format(case_id))
        if not os.path.exists(mhd_path):
            print('not exists {}'.format(case_id))
            continue
        image, origin, spacing = load_itk(mhd_path)
        mask_image = np.zeros_like(image, np.uint8) # zyx
        if case_id in annotation_obj.keys():
            for case_annotation in annotation_obj[case_id]:
                # 针对每一个病灶
                start_coord = [
                    case_annotation['coordZ'] - case_annotation['diameterZ'] / 2.,
                    case_annotation['coordY'] - case_annotation['diameterY'] / 2.,
                    case_annotation['coordX'] - case_annotation['diameterX'] / 2.


                ]
                end_coord = [
                    case_annotation['coordZ'] + case_annotation['diameterZ'] / 2.,
                    case_annotation['coordY'] + case_annotation['diameterY'] / 2.,
                    case_annotation['coordX'] + case_annotation['diameterX'] / 2.
                ]

                start_pixel = world2pixel(start_coord, spacing, origin)
                end_pixel = world2pixel(end_coord, spacing,  origin)
                # print(start_pixel)
                # print(end_pixel)
                mask_image[
                start_pixel[0]:end_pixel[0]+1,
                start_pixel[1]:end_pixel[1]+1,
                start_pixel[2]:end_pixel[2]+1] = case_annotation['label']
        save_path = os.path.join(save_dir, '{}.mhd'.format(case_id))
        save_mhd_image(mask_image, save_path)


if __name__ == '__main__':
    # dataset_dir = '/Users/liang/Documents/datasets/chestCT'
    dataset_dir = '/mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT'
    annotation_path = os.path.join(dataset_dir, 'chestCT_round1_annotation.csv')
    image_dir = os.path.join(dataset_dir, 'train')
    annotation_dir = os.path.join(dataset_dir, 'annotation')
    if not os.path.exists(annotation_dir):
        os.mkdir(annotation_dir)
    generate_masks(annotation_path, image_dir, annotation_dir)
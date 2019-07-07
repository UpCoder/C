# -*- coding=utf-8 -*-
import SimpleITK as sitk
import numpy as np
import csv
from dataset import config


def resolve_annotation(annotation_path):
    '''
    解析annotation
    :param annotation_path:
    :return: 返回一个dict，key是ID，Value 是list，每个元素代表一个object， object是字典类型数据 key为coordX，coordY，coordZ，W，H，Z，label
    '''
    return_obj = {}
    with open(annotation_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        count = 0
        for row in reader:
            if count == 0:
                count += 1
                continue
            cur_id = row[0]
            cur_coord_x = row[1]
            cur_coord_y = row[2]
            cur_coord_z = row[3]
            cur_size_x = row[4]
            cur_size_y = row[5]
            cur_size_z = row[6]
            cur_label = row[7]
            if cur_id in return_obj.keys():
                return_obj[cur_id].append(
                    {
                        'coordX': float(cur_coord_x),
                        'coordY': float(cur_coord_y),
                        'coordZ': float(cur_coord_z),
                        'diameterX': float(cur_size_x),
                        'diameterY': float(cur_size_y),
                        'diameterZ': float(cur_size_z),
                        'label': config.label_mapping[cur_label]

                    }
                )
            else:
                return_obj[cur_id] = []
                return_obj[cur_id].append(
                    {
                        'coordX': float(cur_coord_x),
                        'coordY': float(cur_coord_y),
                        'coordZ': float(cur_coord_z),
                        'diameterX': float(cur_size_x),
                        'diameterY': float(cur_size_y),
                        'diameterZ': float(cur_size_z),
                        'label': config.label_mapping[cur_label]

                    }
                )
    return return_obj


def load_itk(filename):
    '''
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    # tmp = itkimage.GetOrigin()
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    # z,y,x
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def save_mhd_image(image, file_name):
    header = sitk.GetImageFromArray(image)
    sitk.WriteImage(header, file_name)


def world2pixel(coords, spacing, origin):
    '''
    将现实世界的坐标转化为pixel级别
    :param coords:
    :param spacing:
    :param origin:
    :return:
    '''
    if len(coords) != len(spacing) or len(coords) != len(origin):
        assert False
    locations = []
    for idx, coord in enumerate(coords):
        dis_diff = float(coord - origin[idx])
        locations.append(
            int(dis_diff / spacing[idx])
        )
    return locations


if __name__ == '__main__':
    # file_name = '/Users/liang/Documents/datasets/chestCT/train_part1/318818.mhd'
    # load_itk(filename=file_name)
    annotation_file_path = '/Users/liang/Documents/datasets/chestCT/chestCT_round1_annotation.csv'
    obj = resolve_annotation(annotation_file_path)
    print len(obj.keys())
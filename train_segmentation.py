import argparse
from models import UNet
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str, default='UNet')
    parser.add_argument('-b', '--backbone', type=str, default='resnet50')
    parser.add_argument('-nc', '--num_classes', type=int, default=1)
    parser.add_argument('-d', '--dataset_dir', type=str,
                        default='/mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/tfrecords/V2')
    parser.add_argument('-td', '--train_dir', type=str,
                        default='/mnt/cephfs_hl/vc/liangdong.tony/models/ChestCTChallenge/ck')
    parser.add_argument('-ld', '--log_dir', type=str,
                        default='/mnt/cephfs_hl/vc/liangdong.tony/models/ChestCTChallenge/logs')
    args = vars(parser.parse_args())
    train_dir = args['train_dir']
    train_dir = os.path.join(train_dir, 'seg_dataset_{}_{}_{}'.format(os.path.basename(args['dataset_dir']),
                                                                      args['network'], args['backbone']))
    log_dir = args['log_dir']
    log_dir = os.path.join(log_dir, 'seg_dataset_{}_{}_{}'.format(os.path.basename(args['dataset_dir']),
                                                                  args['network'], args['backbone']))

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if args['network'] == 'UNet':
        unet_obj = UNet.UNet(backbone=args['backbone'], num_classes=1, encoder_weights='imagenet',
                             train_dir=train_dir, log_dir=log_dir, dataset_dir=args['dataset_dir'])
        unet_obj.train()
        # python ./train_segmentation.py --network=UNet --backbone=resnet50 --num_classes=1 --train_dir=/mnt/cephfs_hl/vc/liangdong.tony/models/ChestCTChallenge/ck --logs=/mnt/cephfs_hl/vc/liangdong.tony/models/ChestCTChallenge/logs

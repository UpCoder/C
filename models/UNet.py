import sys
sys.path.insert(0, '.')
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from dataset.load_datasets import load_dataset
import keras
import config as global_config
from dataset import config as dataset_config
import os
import tensorflow as tf
from models.callbacks import Tensorboard, CustomCheckpointer
os.environ["CUDA_VISIBLE_DEVICES"] = global_config.gpu_ids
gpu_config = tf.ConfigProto(allow_soft_placement=True)
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
keras.backend.set_session(sess)
dataset_dir = '/mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/tfrecords/V1'


class UNet:
    def __init__(self, backbone='resnet50', num_classes=1, encoder_weights='imagenet'):
        self.preprocess_input = get_preprocessing(backbone)
        self.input_image, _, _, self.input_mask = load_dataset(dataset_dir)
        self.input_image = self.preprocess_input(self.input_image)
        print('input_image is ', self.input_image, self.input_mask)
        with tf.device('/cpu:0'):
            self.model = Unet(backbone, num_classes=num_classes, encoder_weights=encoder_weights,
                              activation='sigmoid', input_shape=dataset_config.input_shape)
        print('output are ', self.model.outputs)
        summary_op = tf.summary.merge_all()
        cb_tensorboard = Tensorboard(summary_op, batch_interval=10, log_dir='./')
        cb_checkpointer = CustomCheckpointer('./', self.model, monitor='loss', mode='min',
                                             save_best_only=False, verbose=False)
        self.cbs = [cb_tensorboard, cb_checkpointer]
        self.parallel_model = keras.utils.multi_gpu_model(self.model, gpus=global_config.num_gpus)
        self.parallel_model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

    def train(self):
        print(dataset_config.num_total_samples // dataset_config.batch_size)
        self.parallel_model.fit(
            x=self.input_image,
            y=self.input_mask,
            steps_per_epoch=dataset_config.num_total_samples // dataset_config.batch_size, epochs=10,
            callbacks=self.cbs
        )


if __name__ == '__main__':
    # encoder_weights = '/mnt/cephfs_hl/vc/liangdong.tony/PycharmProjects/ChestCTChallenge/' \
    #                   'pretrained_models/resnet50_imagenet_1000_no_top.h5'
    encoder_weights = 'imagenet'
    unet_model = UNet(encoder_weights=encoder_weights)
    unet_model.train()
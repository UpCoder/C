#!/usr/bin/env bash
nohup wget http://tianchi-media.oss-cn-beijing.aliyuncs.com/231724_guangxi/chestCT_round1_train_part1.zip -O /mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/chestCT_round1_train_part1.zip > ./logs/dataset/download_train_part1.log &
nohup wget http://tianchi-media.oss-cn-beijing.aliyuncs.com/231724_guangxi/chestCT_round1_train_part2.zip -O /mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/chestCT_round1_train_part2.zip > ./logs/dataset/download_train_part2.log &
nohup wget http://tianchi-media.oss-cn-beijing.aliyuncs.com/231724_guangxi/chestCT_round1_train_part3.zip -O /mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/chestCT_round1_train_part3.zip > ./logs/dataset/download_train_part3.log &
nohup wget http://tianchi-media.oss-cn-beijing.aliyuncs.com/231724_guangxi/chestCT_round1_train_part4.zip -O /mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/chestCT_round1_train_part4.zip > ./logs/dataset/download_train_part4.log &
nohup wget http://tianchi-media.oss-cn-beijing.aliyuncs.com/231724_guangxi/chestCT_round1_train_part5.zip -O /mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/chestCT_round1_train_part5.zip > ./logs/dataset/download_train_part5.log &
nohup wget http://tianchi-media.oss-cn-beijing.aliyuncs.com/231724_guangxi/chestCT_round1_annotation.csv -O /mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/chestCT_round1_annotation.csv > ./logs/dataset/download_annotation.log &
nohup wget http://tianchi-media.oss-cn-beijing.aliyuncs.com/231724_guangxi/chestCT_round1_testA.zip -O /mnt/cephfs_hl/vc/liangdong.tony/datasets/chestCT/chestCT_round1_testA.zip > ./logs/dataset/download_testA.log &
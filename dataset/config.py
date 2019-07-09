# -*- coding=utf-8 -*-
label_mapping = {
    '1': 1, # 结节
    '5': 2, # 索条
    '31': 3,    # 动脉硬化或钙化
    '32': 4,    # 淋巴结钙化
}
num_parallel_reader = 16
batch_size = 30
num_prefecth = 1
shuffle_size = 300
input_shape = (512, 512, 3)
num_total_samples = 100000

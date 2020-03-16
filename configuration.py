#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 loop     Huazhong University of Science and Technology

# Config file for net and train

TRAIN_CONFIG = {
    'profile': False,
    'train_image_shape': (640, 640),
    'data_format': 'NHWC',
    'use_rotation': False,
    'm': 0.4,
    'n': 2,
    'OHM': True,

    'weight_decay': 5e-4,

    'batch_size': 6,

    'log_dir': 'Logs',

    'epoch': 600,

    'lr_config': {
        'lr_boundaries': [200, 400],
        'lr_values': [1e-3, 1e-4, 1e-5]
    },
    'deformable': False,
    # loss weights
    # 'complete_weight': 1,
    'shrink_weight': 5,
    'binary_weight': 1,
    'thresh_weight': 10,
    'k': 50,

    # dataset name
    'data_name': 'icdar2015',
    'data_config': {
        'read_num_p': 2,
        'para_num_p': 2,
        'pro_num_p': 4,
        'buffer_size': 100,
        'prefetch': 50
    },

    'min_size': 640,
    'ran_scale': [0.5, 1.0, 2.0, 3.0]
}

TEST_CONFIG = {
    'id': '0',
    'ckpt': '',

    'log_dir': 'Logs',
    # TODO your path here
    'test_dir': '/home/keyan/NewDisk/ZhangXiong/text_detection/psenet/ICDAR2015/test_images',

    'n': TRAIN_CONFIG['n'],
    'threshold_kernel': 0.3,
    'threshold': 0.3,
    'aver_score': 0.7,

    'image_size': {
        'w': 1280,   # 2240
        'h': 768,   # 1280
        'fixed_size': False,
        'scale': 2
    }
}

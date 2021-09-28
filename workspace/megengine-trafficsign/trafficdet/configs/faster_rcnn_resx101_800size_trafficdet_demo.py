# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine import hub
import models

@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "faster_rcnn_res101_coco_3x_800size_42dot6_2538b0ff.pkl"
)

class CustomerConfig(models.FasterRCNNConfig):
    def __init__(self):
        super().__init__()
        
        # ------------------------ dataset cfg ---------------------- #
        self.train_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/train.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/val.json",
            test_final_ann_file="annotations/test.json",
            remove_images_without_annotations=False,
        )
        self.backbone = "resnext101_32x8d"
        self.num_classes = 5
        
        self.train_image_short_size = (600,736,800,932,1080,1200,1333,1500)
        self.train_image_max_size = 2000
        
        self.test_image_short_size = 1600
        self.test_image_max_size = 2000
        self.test_max_boxes_per_image = 1000
        self.test_prev_nms_top_n = 2000
        self.test_cls_threshold = 0.0001

        
        # ------------------------ training cfg ---------------------- #
        self.basic_lr = 0.02 / 16 * 2
        self.max_epoch = 20
        self.lr_decay_stages = [12, 16]
        self.nr_images_epoch = 2226
        self.warm_iters = 100
        self.log_interval = 100


Net = models.FasterRCNN
Cfg = CustomerConfig


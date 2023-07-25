#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch
import torch.distributed as dist

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 256)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 256)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False
        
        self.data_dir = "datasets"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_valid.json"
        self.test_ann = "instances_valid.json"
        self.output_dir = "./Yolox_barrels_det"
        self.num_classes = 1

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        """
        Get dataloader according to cache_img parameter.
        Args:
            no_aug (bool, optional): Whether to turn off mosaic data enhancement. Defaults to False.
            cache_img (str, optional): cache_img is equivalent to cache_type. Defaults to None.
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
                None: Do not use cache, in this case cache_data is also None.
        """
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            if self.cache_dataset is None:
                assert cache_img is None, "cache is True, but cache_dataset is None"
                dataset = COCODataset(
                    data_dir=self.data_dir,
                    json_file=self.train_ann,
                    img_size=self.input_size,
                    preproc=TrainTransform(
                        max_labels=50,
                        flip_prob=self.flip_prob,
                        hsv_prob=self.hsv_prob),
                    cache=False,
                    cache_type=cache_img,
                    name="train"
                )
            else:
                dataset = self.cache_dataset

        self.dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader
    
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="valid" if not testdev else "test",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
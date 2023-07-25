#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import json
import os
from glob import glob
from copy import deepcopy


if __name__ == '__main__':
    json_filename = "/home/lutao/datasets/ocr/ICDAR_LSVT/train_full_labels.json"
    output_filename = "/home/lutao/datasets/ocr/ICDAR_LSVT/train_full_labels_tr.json"
    
    root_dir = "/home/lutao/datasets/ocr/ICDAR_LSVT/"
    image_dirs = ["test_part1_images", "test_part2_images", "train_full_images_0", "train_full_images_1"]
    image_names_list = []
    # load all image names
    for image_dir in image_dirs:
        a = glob("{}/{}/*.jpg".format(root_dir, image_dir))
        for image_path in a:
            image_name = image_path.split("/")[-1].split(".")[0]
            # print("image_name: {}".format(image_name))
            image_names_list.append(image_name)
            
    print("all images: {}, image-0: {}".format(
        len(image_names_list),
        image_names_list[0]
    ))
    
    with open(json_filename) as f:
        datas: dict = json.load(f)
    datas_copy = {}
    for image_name, items in datas.items():
        if image_name not in image_names_list:
            continue
        a = []
        try:
            for item in items:
                chinese_name = item["transcription"]
                print("image_name: {}, chinese_name: {}".format(image_name, chinese_name))
                chinese_name = bytes(chinese_name, encoding='utf-8')
                # print(type(bytes.decode('utf-8', chinese_name)))
                chinese_name = str(chinese_name, encoding="utf-8")
                # chinese_name = chinese_name.decode('utf-8').encode('utf-8')
                item["transcription"] = chinese_name
                a.append(item)
        except Exception as e:
            continue
        datas_copy[image_name] = a
    
    with open(output_filename, "w") as f:
        json.dump(datas_copy, f, indent=3)

#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
from glob import glob
import os
# import ast
import json


if __name__ == '__main__':
    root_dir = "/home/lutao/datasets/ocr/det_data/ocr_det_data/bankpagedata"
    category = "train"
    FILE_PATH = "{}/{}/".format(root_dir, category) #####
    gt_dir = "{}/{}/{}_label.txt".format(root_dir, category, category)
    images_dir = "{}/{}/images".format(root_dir, category)
    out = {'annotations': [], 
            'categories': [{"id": 1, "name": "txt_area", "supercategory": ""}], ##### change the categories to match your dataset!
            'images': [],
            'info': {"contributor": "", "year": "", "version": "", "url": "", "description": "", "date_created": ""},
            'licenses': {"id": 0, "name": "", "url": ""}
            }
    ann_id_counter = 0
    image_id = 0
    
    show_labels = True
    with open(gt_dir, "r") as f:
        for line in f.readlines():
            image_full_path = str(line).split("\t")[0]
            image_name = image_full_path.split("/")[-1]
            
            image = cv2.imread(os.path.join(images_dir, image_name))
            image_height, image_width = image.shape[0:2]
            # print("full path: {}".format(str(line).split("\t")[1]))
            labels = json.loads(line.split("\t")[1])
            if len(labels) > 0:
                for label in labels:
                    coordiates = label["points"]
                    top_left = coordiates[0]
                    right_bottom = coordiates[2]
                    # x1, y1, x2, y2
                    bbox = [top_left[0], top_left[1], right_bottom[0], right_bottom[1]]
                    bbox_width = right_bottom[0] - top_left[0]
                    bbox_height = right_bottom[1] - top_left[1]
                    bbox_area = bbox_height * bbox_width
                    ann = {'id': ann_id_counter,
                        'bbox': bbox,
                        'area': bbox_area,
                        'iscrowd': 0,
                        'image_id': image_id,
                        'category_id': 1  # +1 if needed    
                    }
                    out['annotations'].append(ann)
                    ann_id_counter = ann_id_counter + 1 
                    if show_labels:
                        cv2.rectangle(image, 
                                    tuple(top_left),
                                    tuple(right_bottom),
                                    (0, 0, 255), 3)
                if show_labels:
                    image = cv2.resize(image, (720, 640))
                    cv2.imshow("image", image)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        break
                    else:
                        continue
            imgs = {'id': image_id,
                "height": image_height, ##### change the 600 to your raw image height
                "width": image_width, ##### change the 800 to your raw image width
                "file_name": image_name,
                "coco_url": "", 
                "flickr_url": "", 
                "date_captured": 0, 
                "license": 0
            }
            image_id += 1
            out['images'].append(imgs)
    
    with open('instances_{}.json'.format(category), 'w') as outfile: ##### change the str to the json file name you want
        json.dump(out, outfile, separators=(',', ':'))
    
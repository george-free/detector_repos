#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
from glob import glob
import os


if __name__ == '__main__':
    images_dir = "/home/lutao/datasets/ocr/det_data/RCTW/train_images/train_images"
    gt_dir = "/home/lutao/datasets/ocr/det_data/RCTW/train_images/train_gts"
    for image_path in glob(images_dir + "/*.jpg"):
        image_name = image_path.split("/")[-1].split(".")[0]
        txt_path = os.path.join(gt_dir, image_name + ".txt")
        
        image = cv2.imread(image_path)
        image_shape = image.shape[0:2]
        with open(txt_path, "r") as f:
            for line in f.readlines():
                # print("line: {}".format(str(line)))
                coordiates = str(line).split(",")[0:-2]
                print("coordinate: {}".format(coordiates))
                coordiates = list(map(int, coordiates))
                top_left = coordiates[0:2]
                right_bottom = coordiates[4:6]
                # print("top: {}".format())
                cv2.rectangle(image, 
                              tuple(top_left),
                              tuple(right_bottom),
                              (0, 0, 255),
                              3)
                cv2.line(image, tuple(coordiates[0:2]), tuple(coordiates[2:4]), (255, 0, 0), 2)
                cv2.line(image, tuple(coordiates[2:4]), tuple(coordiates[4:6]), (255, 0, 0), 2)
                cv2.line(image, tuple(coordiates[4:6]), tuple(coordiates[6:8]), (255, 0, 0), 2)
                cv2.line(image, tuple(coordiates[6:8]), tuple(coordiates[0:2]), (255, 0, 0), 2)
        
        image = cv2.resize(image, (640, 480))
        cv2.imshow("image", image)
        key = cv2.waitKey(0)
        if key == 'q':
            break
        else:
            continue
        
                
        
    
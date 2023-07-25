#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from glob import glob
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import math
from copy import deepcopy


def visualize_label_ele():
    images_dir = "/home/lutao/datasets/pd/images"
    labels_dir = "/home/lutao/datasets/pd/xml"
    output_dir = "/home/lutao/datasets/pd/images_by_labels"
    os.makedirs(output_dir)
    # label_dict = {
    #     "010101021": "纵向、横向裂纹-1",
    #     "010101022": "纵向、横向裂纹-2",
    #     "010101023": "纵向、横向裂纹-3",
    #     "010101031": "法兰杆法兰锈蚀-1",
    #     "010101031": "法兰杆法兰锈蚀-2",
    #     "010101061": "异物、鸟巢",
    #     "010101071": "塔顶损坏",
    # }
    preview_size = (960, 720)
    object_labels_list = []
    
    for label_file in glob(labels_dir + "/*.xml"):
        content_tree = ET.parse(label_file)
        img_filename = content_tree.find("filename").text
        img_path = os.path.join(images_dir, img_filename)
        image = cv2.imread(img_path)
        objects_info = {}
        
        for obj in content_tree.findall("object"):
            id_code = obj.find("code").text
            obj_name = obj.find("DeDescription").text
            obj_level = obj.find("DefectLevel").text
            bbox = obj.find("bndbox")
            bbox_xyxy = [
                math.floor(float(bbox.find("xmin").text)),
                math.floor(float(bbox.find("ymin").text)),
                math.floor(float(bbox.find("xmax").text)),
                math.floor(float(bbox.find("ymax").text))
            ]
            if obj_name not in objects_info:
                objects_info[obj_name] = []
            objects_info[obj_name].append(bbox_xyxy)
            # draw labels and bounding boxes
            # image_show = deepcopy(image)
            # image = cv2.rectangle(
            #     image,
            #     (bbox_xyxy[0], bbox_xyxy[1]),
            #     (bbox_xyxy[2], bbox_xyxy[3]),
            #     (0, 0, 255),
            #     10)
            print("name: {}, level: {}".format(obj_name, obj_level))
            # font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
            # cv2.putText(image, 
            #             obj_name, 
            #             (bbox_xyxy[0], bbox_xyxy[1]), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2,
            #             cv2.LINE_AA, False)
            # cv2.putText(image, 
            #             obj_level, 
            #             (bbox_xyxy[0], bbox_xyxy[1] + 60), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

            # image_show = cv2.resize(image, preview_size)
            # cv2.imshow("image", image_show)
            # key = cv2.waitKey(0) & 0xFF
            # if key == ord("q"):
            #     break
            # else:
            #     continue
            # if obj_name not in object_labels:
            #     os.makedirs(os.path.join(output_dir, obj_name))
            #     object_labels.append(obj_name)
            # output_image_path = os.path.join(output_dir, obj_name, img_filename)
            # cv2.imwrite(output_image_path, image_show)
        image_ori = cv2.resize(image, preview_size, interpolation=cv2.INTER_CUBIC)
        for tag, objs in objects_info.items():
            image_show = deepcopy(image)
            mask = np.zeros(image_show.shape[0:2], dtype=np.uint8)
            for bbox_xyxy in objs:
                # image_show = cv2.rectangle(
                # image_show,
                # (bbox_xyxy[0] - 2, bbox_xyxy[1] - 2),
                # (bbox_xyxy[2] + 2, bbox_xyxy[3] + 2),
                # (0, 0, 255),
                # 6)
                mask[bbox_xyxy[1]: bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]] = 1
            # image_show[..., 0][mask > 0] = 255
            image_show = cv2.resize(image_show, preview_size, interpolation=cv2.INTER_CUBIC)
            img_output = np.hstack([image_show, image_ori])
            if tag not in object_labels_list:
                object_labels_list.append(tag)
                os.makedirs(os.path.join(output_dir, tag))
            output_image_path = os.path.join(output_dir, tag, img_filename)
            cv2.imwrite(output_image_path, img_output)

if __name__ == "__main__":
    # visualize_label_ele()
    mask = np.zeros((720, 960), dtype=np.uint8)
    mask[147:414, 600:835] = 255
    cv2.imwrite("mask.png", mask)

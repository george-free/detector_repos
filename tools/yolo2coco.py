import os
import numpy as np
import json
import cv2
from glob import glob

category = "train"
FILE_PATH = '/home/lutao/datasets/face_mask_data/face_mask_v0/{}/'.format(category) #####
out = {'annotations': [], 
           'categories': [{"id": 1, "name": "face_without_mask", "supercategory": ""}, 
                          {"id": 2, "name": "face_with_mask", "supercategory": ""},
                          {"id": 3, "name": "face_with_partial_mask", "supercategory": ""}], ##### change the categories to match your dataset!
           'images': [],
           'info': {"contributor": "", "year": "", "version": "", "url": "", "description": "", "date_created": ""},
           'licenses': {"id": 0, "name": "", "url": ""}
           }
ann_id_counter = 0
image_id = 0


def convert_data_format(file_dir, file_name:str):
    # id, bbox, iscrowd, image_id, category_id
    global ann_id_counter
    global image_id
    global out
    
    txt_path = os.path.join(file_dir, file_name)
    id = file_name.split('.')[0]
    image_name = id + '.jpg' ##### change '.jpg' to other image formats if the format of your image is not .jpg
    image = cv2.imread(os.path.join(file_dir, image_name))
    image_height, image_width = image.shape[0:2]

    txt = open(txt_path, 'r')
    for line in txt.readlines(): # if txt.readlines is null, this for loop would not run
        data = line.strip()
        data = data.split()
        if len(data) > 0:
            # convert the center into the top-left point!
            data[1] = float(data[1]) * image_width - 0.5 * float(data[3])* image_width ##### change the 800 to your raw image width
            data[2] = float(data[2])* image_height - 0.5 * float(data[4])* image_height ##### change the 600 to your raw image height
            data[3] = float(data[3])* image_width ##### change the 800 to your raw image width
            data[4] = float(data[4])* image_height ##### change the 600 to your raw image height
            bbox = [data[1], data[2], data[3], data[4]]
            ann = {'id': ann_id_counter,
                'bbox': bbox,
                'area': data[3] * data[4],
                'iscrowd': 0,
                'image_id': image_id,
                'category_id': int(data[0])  # +1 if needed    
            }
            out['annotations'].append(ann)
            ann_id_counter = ann_id_counter + 1 
    
    imgs = {'id': image_id,
            'height': image_height, ##### change the 600 to your raw image height
            'width': image_width, ##### change the 800 to your raw image width
            'file_name': image_name,
            "coco_url": "", 
            "flickr_url": "", 
            "date_captured": 0, 
            "license": 0
    }
    image_id += 1
    out['images'].append(imgs)


if __name__ == '__main__':
    files = glob(FILE_PATH + "*.txt")
    files.sort()
    for file in files:
        file_name = file.split('/')[-1]
        convert_data_format(FILE_PATH, file_name)

    with open('instances_merge_{}.json'.format(category), 'w') as outfile: ##### change the str to the json file name you want
        json.dump(out, outfile, separators=(',', ':'))
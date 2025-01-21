"""
https://github.com/xingyizhou/CenterTrack
Modified by Xiaoyu Zhao
https://github.com/MCG-NJU/SportsMOT/blob/main/codes/conversion/mot_to_coco.py

https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/convert_mot_to_coco.py

There are extra many convert_X_to_coco.py

https://cocodataset.org/#format-data
"""
import os
import numpy as np
import json
import cv2
from tqdm import tqdm

DATA_PATH = "datasets/volleyball"
OUT_PATH = os.path.join(DATA_PATH, "annotations")
os.makedirs(OUT_PATH, exist_ok=True)
SPLITS = [str(i) for i in range(55)]
HALF_VIDEO = False
CREATE_SPLITTED_ANN = True
USE_DET = False
CREATE_SPLITTED_DET = False
WIDTH = 1280
HEIGHT = 720

for split in SPLITS:
    data_path = os.path.join(DATA_PATH, "videos", split)
    out_path = os.path.join(OUT_PATH, "{}.json".format(split))
    out = {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [{
            "id": 1,
            "name": "pedestrian"
        }]
    }
    video_list = [os.path.split(os.path.split(f.path)[-2])[-1] + "/" + os.path.split(f.path)[-1] for f in os.scandir(data_path) if f.is_dir()]
    image_cnt = 0
    ann_cnt = 0
    for seq in tqdm(sorted(video_list)):
        if ".DS_Store" in seq:
            continue
        out["videos"].append({"id": seq, "file_name": seq})
        seq_path = os.path.join(data_path, os.path.split(seq)[-1])
        img_path = os.path.join(seq_path)
        # ann_path = os.path.join(seq_path, "gt/gt.txt")
        images = sorted(os.listdir(img_path))
        num_images = len([image for image in images
                          if "jpg" in image])  # half and half

        if HALF_VIDEO and ("half" in split):
            image_range = [0, num_images // 2] if "train" in split else \
                            [num_images // 2 + 1, num_images - 1]
        else:
            image_range = [0, num_images - 1]

        for i in range(num_images):
            if i < image_range[0] or i > image_range[1]:
                continue
            # img = cv2.imread(
            #     os.path.join(data_path,
            #                  "{}/img1/{:06d}.jpg".format(seq, i + 1)))
            # height, width = img.shape[:2]
            image_info = {
                "file_name": "{}/img1/{:06d}.jpg".format(seq,
                                                         i + 1),  # image name.
                "id":
                image_cnt + i + 1,  # image number in the entire training set.
                "frame_id": i + 1 - image_range[
                    0],  # image number in the video sequence, starting from 1.
                "prev_image_id": image_cnt +
                i if i > 0 else -1,  # image number in the entire training set.
                "next_image_id":
                image_cnt + i + 2 if i < num_images - 1 else -1,
                "video_id": seq,
                "height": HEIGHT,
                "width": WIDTH
            }
            out["images"].append(image_info)
        print("{}: {} images".format(seq, num_images))
        
        image_cnt += num_images
    print("loaded {} for {} images and {} samples".format(
        split, len(out["images"]), len(out["annotations"])))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
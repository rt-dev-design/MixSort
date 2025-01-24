"""
https://github.com/xingyizhou/CenterTrack
Modified by Xiaoyu Zhao
https://github.com/MCG-NJU/SportsMOT/blob/main/codes/conversion/mot_to_coco.py

https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/convert_mot_to_coco.py

There are extra many convert_X_to_coco.py

https://cocodataset.org/#format-data
"""
import os
import json
from tqdm import tqdm
import sys

assert len(sys.argv) == 2, "Usage: python convert_dataset_to_coco.py [volleyball | nba]"
dataset_name = sys.argv[1]
assert dataset_name in ["volleyball", "nba"], "only support [volleyball | nba]"

DATA_PATH = f"datasets/{dataset_name}"
OUT_PATH = os.path.join(DATA_PATH, "annotations")
os.makedirs(OUT_PATH, exist_ok=True)

video_set_list = [str(s) for s in sorted([int(os.path.split(f.path)[-1]) for f in os.scandir(f"datasets/{dataset_name}/videos") if f.is_dir()])]
WIDTH = 1280
HEIGHT = 720

out_path = os.path.join(OUT_PATH, "val.json")
out = {
    "images": [],
    "annotations": [],
    "videos": [],
    "categories": [{
        "id": 1,
        "name": "pedestrian"
    }]
}
image_cnt = 0
video_cnt = 0

for video_set in video_set_list:
    data_path = os.path.join(DATA_PATH, "videos", video_set)
    video_list = sorted([int(os.path.split(f.path)[-1]) for f in os.scandir(data_path) if f.is_dir()])
    video_list = [os.path.split(data_path)[-1] + "/" + str(v) for v in video_list]
    for seq in tqdm(video_list, total=len(video_list), desc="video {}".format(video_set)):
        if ".DS_Store" in seq:
            continue
        video_cnt += 1
        out["videos"].append({"id": video_cnt, "file_name": seq})
        seq_path = os.path.join(data_path, os.path.split(seq)[-1])
        img_path = os.path.join(seq_path)
        images = sorted(os.listdir(img_path))
        num_images = len([image for image in images if "jpg" in image])  # half and half
        assert num_images == 41 if dataset_name == "volleyball" else 72, "{} has {} images, which is not right".format(seq, num_images)
        image_range = [0, num_images - 1]

        for i in range(num_images):
            if i < image_range[0] or i > image_range[1]:
                continue
            image_info = {
                "file_name": "{}/{}".format(seq, images[i]),  # image name.
                "id": image_cnt + i + 1,  # image number in the entire training set.
                "frame_id": i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                "prev_image_id": image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                "next_image_id": image_cnt + i + 2 if i < num_images - 1 else -1,
                "video_id": video_cnt,
                "height": HEIGHT,
                "width": WIDTH
            }
            out["images"].append(image_info)
        tqdm.write("clip {}: {} images".format(seq, num_images))
        image_cnt += num_images

with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
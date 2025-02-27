import os
import sys
import cv2
import numpy as np
import tkinter as tk

def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float64)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def txt2display(dataset_path, selection=None, start_from=None):
    print("Starting txt2display")
    color_list = colormap()
    track_results_path = os.path.join(dataset_path, "track_results")
    
    all_videos = sorted([int(clip) for clip in os.listdir(track_results_path)])
    tracks_of_all_clips = []
    for video in all_videos:
        to_sort = [int(file.split(".")[0]) for file in os.listdir(os.path.join(track_results_path, str(video)))]
        track_files = [os.path.join(track_results_path, str(video), str(f) + ".txt") for f in sorted(to_sort)]
        tracks_of_all_clips.extend(track_files)
    
    if selection is not None:  
        selection = [os.path.join(track_results_path, s + ".txt") for s in selection]
    
    if selection is None and start_from is not None:
        start_from_track = os.path.join(track_results_path, start_from + ".txt")
        assert start_from_track in tracks_of_all_clips, "'start_from' is not a valid clip name"
        tracks_of_all_clips = tracks_of_all_clips[tracks_of_all_clips.index(start_from_track):]
    
    is_volleyball = "volleyball" in dataset_path
    is_nba = "nba" in dataset_path
    RESIZE_SCALE = (1280, 720)

    win_num = 0
    only_visualize_the_first_frame_of_each_clip = False
    for track_file in tracks_of_all_clips if selection is None else selection:
        clip_dir = track_file[:-4].replace("track_results", "videos")
        if is_volleyball:
            images = ['place holder for index 0'] + [os.path.join(clip_dir, str(image)) + ".jpg" for image in sorted([int(f[:-4]) for f in os.listdir(clip_dir)])]
        elif is_nba:
            images = ['place holder for index 0'] + [os.path.join(clip_dir, img) for img in sorted([f for f in os.listdir(clip_dir)])]
        else:
            assert False, "This script assumes the dataset is either 'volleyball' or 'nba'"
        
        txt_dict = dict()    
        with open(track_file, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                img_id = linelist[0]
                obj_id = linelist[1]
                bbox = [float(linelist[2]), float(linelist[3]), 
                        float(linelist[2]) + float(linelist[4]), 
                        float(linelist[3]) + float(linelist[5]), int(obj_id)]
                if int(img_id) in txt_dict:
                    txt_dict[int(img_id)].append(bbox)
                else:
                    txt_dict[int(img_id)] = list()
                    txt_dict[int(img_id)].append(bbox)
        # 获取屏幕的宽度和高度
        screen_width = tk.Tk().winfo_screenwidth()
        screen_height = tk.Tk().winfo_screenheight()


        for img_id in sorted(txt_dict.keys()):
            img = cv2.imread(images[img_id])
            img = cv2.resize(img, RESIZE_SCALE)
                
            for bbox in txt_dict[img_id]:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_list[bbox[4]%79].tolist(), thickness=2)
                cv2.putText(img, "{}".format(int(bbox[4])), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_list[bbox[4]%79].tolist(), 2)

            # 计算窗口的左上角坐标使其居中
            x_pos = int((screen_width - img.shape[1]) / 2)
            y_pos = int((screen_height - img.shape[0]) / 2)
            window_name = images[img_id]
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) 
            cv2.setWindowTitle(window_name, window_name)
            cv2.moveWindow(window_name, x_pos, y_pos)
            cv2.imshow(window_name, img)
            cv2.waitKey(0)
            if only_visualize_the_first_frame_of_each_clip:
                break
        print("Done visualizing", track_file)
        win_num += 1
        if win_num >= 100:
            win_num = 0
            cv2.destroyAllWindows()
    print("txt2display Done")


if __name__ == '__main__':
    dataset_path = "/media/ssd_2t/home/zrt/datasets/gar/volleyball"
    selection = None # ['22/24290', '1/9930']  # None, which indicates visualizing all, or a list with the form like ['0/7917', '1/9930'], ['21800909/366', '21800919/389']
    start_from = None # '21800919/691'  '21801078/272'  '22/24290'
    txt2display(dataset_path=dataset_path, selection=selection, start_from=start_from)

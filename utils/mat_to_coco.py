import numpy as np
import scipy.io as si
import os
import json
import time
import argparse
import cv2
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Generate COCO format annotation for .mat annotations')
    parser.add_argument('--mat', help='path to the annotation file', type=str)
    parser.add_argument('--image_dir', help='path to the image directory', type=str)

    args = parser.parse_args()
    return args

def load_annotation(mat):
    annotations = si.loadmat(mat)
    print(annotations["RELEASE"])
    return annotations

def save_annotation(annotations, mat):
    directory = os.path.dirname(mat)
    file_name = os.path.basename(mat).split('.', 1)[0] + "_coco"
    with open(os.path.join(directory, file_name + ".json"), 'w') as f:
        json_str = json.dumps(annotations)
        f.write(json_str)

def mat2coco(mat, directory):
    json_dict = {"licenses":[{"name":"","id":0,"url":""}],"info":{"contributor":"","date_created":"","description":"","url":"","version":"","year":""},"categories":[{"id":1,"name":"Human","supercategory":"","keypoints":["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17"],"skeleton":[[6,12],[2,4],[1,2],[13,7],[3,1],[15,17],[6,8],[12,13],[8,10],[4,6],[5,7],[7,9],[5,3],[13,15],[7,6],[14,12],[9,11],[16,14]]}],
    "images": [], "annotations": []}

    joints = torch.from_numpy(load_annotation(mat)["joints"])
    joints = joints.permute(2,0,1)

    for i,file in tqdm(enumerate(sorted(os.listdir(directory)))):
        filename = os.fsdecode(file)

        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            img = cv2.imread(image_path)
            image = {"id": i,
                    "width": img.shape[0],
                    "height": img.shape[1],
                    "file_name": filename,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0}

            kpts = np.array([[max(0.0,x), max(0.0,y), v] for x,y,v in joints[i]]).flatten()
            annotation = {"id":i,
                        "image_id":i,
                        "category_id": 1,
                        "segmentation": [],
                        "area": img.shape[0]*img.shape[1],
                        "bbox":[int(min(kpts[0::3])),int(min(kpts[1::3])),int(max(kpts[0::3]) - min(kpts[0::3])),int(max(kpts[1::3]) - min(kpts[1::3]))],
                        "iscrowd": 0,
                        "attributes": {
                            "occluded": False,
                            "keyframe": False},
                        "keypoints": kpts.tolist(),
                        "num_keypoints": int(sum(kpts[2::3] > 0)) }

            
            json_dict["images"].append(image)
            json_dict["annotations"].append(annotation)

    save_annotation(json_dict, mat)
    print("Done")


if __name__  == "__main__":
    args = parse_args()
    mat2coco(args.mat, args.image_dir)
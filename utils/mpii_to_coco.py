import json
import torch
import numpy as np
import os
from tqdm import tqdm
import cv2

# Mapping : coco -> mpii
COCO_to_MPII = {
    6: 12, 5: 13, 8: 11,
    7: 14, 10: 10, 9: 15,
    12: 2, 11: 3, 14: 1,
    13: 4, 16: 0, 15: 5
}

COCO_to_leed = {
    6: 9, 5: 8, 8: 10,
    7: 7, 10: 11, 9: 6,
    12: 3, 11: 2, 14: 4,
    13: 1, 16: 5, 15: 0
}

COCO_flip = {i:i+1 for i in range(1,17,2)}

def convert_keypoints_mpii_to_coco(mpii_kpts):
    """
    Convertit une liste de 16*3 keypoints MPII vers 17*3 keypoints COCO.
    """
    coco_kpts = [[0, 0, 0] for _ in range(17)]

    for coco_idx, mpii_idx in COCO_flip.items():
        # mpii_kpts est une liste de 48 valeurs (16*3)
        x = mpii_kpts[mpii_idx * 3]
        y = mpii_kpts[mpii_idx * 3 + 1]
        v = mpii_kpts[mpii_idx * 3 + 2]
        if coco_idx < len(coco_kpts):
            coco_kpts[coco_idx] = [x, y, v]

    return [item for kpt in coco_kpts for item in kpt]

def convert_annotations_file(input_json_path, output_json_path):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    for ann in data["annotations"]:
        if "keypoints" in ann and len(ann["keypoints"]) == 16 * 3:
            ann["keypoints"] = convert_keypoints_mpii_to_coco(ann["keypoints"])
            ann["num_keypoints"] = sum(1 for i in range(0, 51, 3) if ann["keypoints"][i+2] > 0)

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Fichier converti sauvegardé dans : {output_json_path}")

def prepare_annotations_file(input_json_path, image_dir):
    json_dict = {"licenses":[{"name":"","id":0,"url":""}],"info":{"contributor":"","date_created":"","description":"","url":"","version":"","year":""},"categories":[{"id":1,"name":"Human","supercategory":"","keypoints":["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17"],"skeleton":[[6,12],[2,4],[1,2],[13,7],[3,1],[15,17],[6,8],[12,13],[8,10],[4,6],[5,7],[7,9],[5,3],[13,15],[7,6],[14,12],[9,11],[16,14]]}],
    "images": [], "annotations": []}

    with open(input_json_path, "r") as f:
        data = json.load(f)

    for i,image in tqdm(enumerate(data)):
        image_path = os.path.join(image_dir, image["image"])
        img = cv2.imread(image_path)
        joints = image["joints"]
        joints_vis = image["joints_vis"]

        image = {"id": i,
                    "width": img.shape[0],
                    "height": img.shape[1],
                    "file_name": image["image"],
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0}

        kpts = np.array([[max(0.0,joint[0]), max(0.0,joint[1]), joints_vis[i]] for i,joint in enumerate(joints)]).flatten()
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

    save_annotation(json_dict, input_json_path)
    print("Done")


def save_annotation(annotations, mat):
    directory = os.path.dirname(mat)
    file_name = os.path.basename(mat).split('.', 1)[0] + "_formatted"
    with open(os.path.join(directory, file_name + ".json"), 'w') as f:
        json_str = json.dumps(annotations)
        f.write(json_str)


if __name__ == "__main__":
    # prepare_annotations_file("../../../Datasets/mpii_human_pose_v1/annotations/person_keypoints_train.json", "../../../Datasets/mpii_human_pose_v1/images/train/images")
    convert_annotations_file(
    "../../../Datasets/leeds-sport-pose-main/annotations/person_keypoints_train.json",
    "../../../Datasets/leeds-sport-pose-main/annotations/person_keypoints_train_coco.json")
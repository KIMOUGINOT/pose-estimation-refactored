import os
import json
import cv2
import argparse
import shutil
import zipfile
from ultralytics import YOLO
from tqdm import tqdm
from time import time
from PIL import Image
from extract_player import detect_and_crop_players

def estimate_pose(input_dir, yolo_model) :
    """Estimate the human pose in the image of the input directory applying a yolo model

    Args:
        image_dir (_str_): Directory of the dataset
        yolo_labels_dir (_str_): Yolo model's name
    """

    start_time = time()
    # Create the directory for images label in Pose_yolo_labels
    yolo_labels_dir = "./Pose_yolo_labels"
    os.makedirs(yolo_labels_dir, exist_ok=True)

    # Fetch the yolo model
    yolo_path = os.path.join("./model", yolo_model)
    model = YOLO(yolo_path)

    # Fetch images
    image_dir = os.path.join("./players", input_dir)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    results = model.predict(source=image_dir, verbose=True, save=False, save_txt=True, project=os.path.join(yolo_labels_dir, input_dir))

    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Pose estimation finished in {elapsed_time:.3f} sec. ")

def preannotation_pipeline(input_dir, yolo_model, dataset_fraction=None) :
    """
    Detect, crop and estimate pose from human. Transform YOLO predictions into COCO and create a zip compliant with CVAT.

    Arguments:
        input_dir (str): Name of the directory which contain the images.
        yolo_model (str): Yolo model used name.
        dataset_fraction (str): Tells if images belong to train or valid or test. 
    """
    start_time = time()
    yolo_model_tracking = "./model/Player-Detection-YOLOv11X-2024-12.pt"
    dataset_path = os.path.join("./raw", input_dir)
    output_path = os.path.join("./players", input_dir)
    yolo_labels_dir = os.path.join("./Pose_yolo_labels", input_dir, "predict", "labels")
    if dataset_fraction is None :
        output_json_path = os.path.join("./pre_annotated", input_dir, "annotations", "person_keypoints.json")
    else : 
        output_json_path = os.path.join("./pre_annotated", input_dir, "annotations", "person_keypoints_"+dataset_fraction+".json")
    output_zip_path = os.path.join("./pre_annotated", f"{input_dir}.zip")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    os.makedirs(os.path.join("./pre_annotated", input_dir, "images"), exist_ok=True)

    detect_and_crop_players(dataset_path, output_path, yolo_model_tracking)
    estimate_pose(input_dir, yolo_model)

    # COCO complete structure
    coco_gt = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "Dataset converted from YOLO to COCO",
            "url": "",
            "version": "",
            "year": 2024
        },
        "categories": [
            {
                "id": 1,
                "name": "Human",
                "supercategory": "",
                "keypoints": [
                    "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "10", "11", "12", "13", "14", "15", "16", "17", "18"
                ],
                "skeleton": [
                    [6, 12], [2, 4], [1, 2], [13, 7], [3, 1], [15, 17], 
                    [6, 8], [12, 13], [8, 10], [4, 6], [5, 7], [7, 9], 
                    [5, 3], [13, 15], [7, 6], [14, 12], [9, 11], [16, 14]
                ]
            }
        ],
        "images": [],
        "annotations": []
    }

    yolo_to_coco_mapping = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    image_id_map = {}  
    annotation_id = 1  

    # yolo label files
    for filename in sorted(os.listdir(yolo_labels_dir)):
        if not filename.endswith(".txt"):
            continue

        image_name = filename.replace(".txt", ".jpg") 
        image_path = os.path.join(output_path, image_name)

        if not os.path.exists(image_path):
            print(f"Image {image_name} not found, ignored.")
            continue

        img = Image.open(image_path)
        width, height = img.size

        #image_id
        if image_name not in image_id_map:
            image_id = len(image_id_map) + 1
            image_id_map[image_name] = image_id

            # add image to COCO
            coco_gt["images"].append({
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0
            })
            shutil.copy(image_path, os.path.join("./pre_annotated", input_dir, "images", image_name))

        image_id = image_id_map[image_name]

        file_path = os.path.join(yolo_labels_dir, filename)
        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            values = line.strip().split()
            if len(values) < 2 + 3 * 18:  # make sure there are at max 18 keypoints
                print(f"Ignored line in {filename}, incorrect format")
                continue

            class_id = int(values[0])
            keypoints = list(map(float, values[2:]))  

            # Convert yolo into coco
            keypoints_abs = []
            for i in yolo_to_coco_mapping:
                x = keypoints[i * 3] * width
                y = keypoints[i * 3 + 1] * height
                visibility = int(round(keypoints[i * 3 + 2] * 2))  # Convert 0/1 in 0/2
                keypoints_abs.extend([x, y, visibility])

            # bbox around keypoints
            x_coords = [keypoints_abs[i] for i in range(0, len(keypoints_abs), 3)]
            y_coords = [keypoints_abs[i + 1] for i in range(0, len(keypoints_abs), 3)]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = bbox[2] * bbox[3]  

            coco_gt["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1, 
                "segmentation": [],  
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "attributes": {
                    "occluded": False,
                    "keyframe": False
                },
                "keypoints": keypoints_abs,
                "num_keypoints": len([v for v in keypoints_abs[2::3] if v > 0])
            })
            annotation_id += 1

    # Save coco annotations
    with open(output_json_path, "w") as f:
        json.dump(coco_gt, f, indent=4)

    # .zip creation
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for foldername, subfolders, filenames in os.walk(os.path.join("./pre_annotated", input_dir)):
            for filename in filenames:
                filepath = os.path.join(foldername, filename)
                zipf.write(filepath, os.path.relpath(filepath, "./pre_annotated"))

    print(f".zip file created in {output_zip_path}")
    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Pre-annotation total : {elapsed_time:.3f} sec. ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-annotate data.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input images directory.")
    parser.add_argument("--model", type=str, required=False, help="Yolo model for detection (.pt).")
    parser.add_argument("--dataset", type=str, required=True, help="Fraction of the dataset (train, valid or test).")

    args = parser.parse_args()

    preannotation_pipeline(args.input_dir, args.model, args.dataset)
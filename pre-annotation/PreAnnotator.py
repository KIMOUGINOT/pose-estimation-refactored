"""
Script to create a pseudo-annotated dataset.

Usage:
  python -m pre-annotation.PreAnnotator --input_dir image_to_annotate/ --output_dir TO_CVAT/ \
  --detection Player-Detection-YOLOv11X-2024-12.pt --pose xpose_resnet18.pt [--keep_full_image] (if you want coordinates \
  in image reference and no crop)
"""

from tqdm import tqdm
import numpy as np
import cv2
import json
import os
import torchvision.transforms as transforms
from PIL import Image

from .PoseEstimator import TorchEstimator
from .PlayerDetector import YOLODetector

class PreAnnotator():

    def __init__(self, image_folder, output_folder, player_detector, pose_estimator, keep_full_image=False):
        self.padding = 40 # padding to add when cropping images
        self.keep_full_image = keep_full_image
        self.directory_name = os.path.basename(image_folder)
        self.player_detector = player_detector
        self.pose_estimator = pose_estimator

        self.image_folder = image_folder
        self.output_folder = os.path.join(output_folder, self.directory_name)
        self.image_output_folder = os.path.join(self.output_folder, "images")
        self.annotations_output_folder = os.path.join(self.output_folder, "annotations")
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.image_output_folder, exist_ok=True)
        os.makedirs(self.annotations_output_folder, exist_ok=True)
        
        print(f"\nDirectory created:")
        print(f"{output_folder}/ ")
        print(f"└── {self.directory_name}/")
        print(f"    ├── images/")
        print(f"    └── annotations/\n") 

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        if image is None:
            print(f"Error loading {image_path}")
            return None

        players = self.player_detector.detect_players(image)
        if players is None or len(players) == 0:
            print(f"No person detected in {image_path}")
            return None

        annotations = []
        filename, file_extension = os.path.splitext(image_path)
        image_basename = os.path.basename(filename)
        image_saved = False  # Avoid duplicating

        for i, (x1, y1, x2, y2) in enumerate(players):
            crop = image[max(0, y1 - self.padding):min(h, y2 + self.padding),
                        max(0, x1 - self.padding):min(w, x2 + self.padding)]
            tensor = transforms.ToTensor()(Image.fromarray(crop))
            preds = self.pose_estimator.estimate_pose(tensor)
            preds = preds[:].flatten()

            keypoints = []
            for k in range(0, len(preds), 2):
                x_kpt = preds[k]
                y_kpt = preds[k+1]
                visible = 2 if (x_kpt > 1e-4 and y_kpt > 1e-4) else 0

                if self.keep_full_image:
                    x_kpt += max(0, x1 - self.padding)
                    y_kpt += max(0, y1 - self.padding)

                keypoints.extend([x_kpt, y_kpt, visible])

            if self.keep_full_image:
                if not image_saved:
                    image_path_to_save = os.path.join(self.image_output_folder, f"{image_basename}{file_extension}")
                    cv2.imwrite(image_path_to_save, image)
                    image_saved = True
                    image_shape = image.shape
                image_name = f"{image_basename}{file_extension}"
            else:
                image_name = f"{image_basename}_{i}{file_extension}"
                image_path_to_save = os.path.join(self.image_output_folder, image_name)
                cv2.imwrite(image_path_to_save, crop)
                image_shape = crop.shape

            annotations.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "keypoints": keypoints,
                "image_name": image_name,
                "image_shape": image_shape
            })

        return annotations


    def process_all_images(self):
        """
        Process all images in the input directory and save COCO-style annotations.
        Handles both cropped images and full original images, depending on config.
        """
        image_metadata = []
        annotations = []
        image_id_map = {}
        annotation_id = 1

        print("Start of image processing...")
        for filename in tqdm(os.listdir(self.image_folder), desc="Detection and pose estimation"):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            image_path = os.path.join(self.image_folder, filename)
            annotation_data = self.process_image(image_path)

            if not annotation_data:
                continue

            for ann in annotation_data:
                image_name = ann["image_name"]
                width, height, _ = ann["image_shape"]

                if image_name not in image_id_map:
                    image_id = len(image_id_map) + 1
                    image_id_map[image_name] = image_id
                    image_metadata.append({
                        "id": image_id,
                        "file_name": image_name,
                        "width": width,
                        "height": height,
                        "license": 0,
                        "flickr_url": "",
                        "coco_url": "",
                        "date_captured": 0
                    })
                else:
                    image_id = image_id_map[image_name]

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Person
                    "bbox": ann["bbox"],
                    "area": ann["bbox"][2] * ann["bbox"][3],
                    "iscrowd": 0,
                    "num_keypoints": sum(1 for v in ann["keypoints"][2::3] if v > 0),
                    "keypoints": ann["keypoints"]
                })
                annotation_id += 1

        print("Image processing ✅")
        print(f"Images saved at {self.image_output_folder}")
        output_json_path = os.path.join(self.annotations_output_folder, "person_keypoints_default.json")
        self.kp_to_json(output_json_path, image_metadata, annotations)


    def kp_to_json(self, output_json_path, image_metadata, annotations):
        """
        Creates a COCO ground truth JSON file with keypoints and bounding boxes.

        Arguments:
            output_json_path (str): Path to save the output JSON file.
            image_metadata (list): List of image dictionaries with COCO-compliant metadata.
            annotations (list): List of annotation dictionaries with COCO keypoints format.
        """

        print(f"Creating COCO Annotation file... ", end='')
        coco_gt = {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {
                "contributor": "SPASH",
                "date_created": "",
                "description": "Dataset for human keypoint annotation ",
                "url": "",
                "version": "",
                "year": 2025
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
            "images": image_metadata,  
            "annotations": annotations  
        }

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        print("✅")

        with open(output_json_path, "w") as f:
            json.dump(coco_gt, f, indent=4)
        print(f"COCO annotation file saved at {output_json_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preannotate a dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input images directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output images directory.")
    parser.add_argument("--detection", type=str, required=True, help="model for detection (.pt).")
    parser.add_argument("--pose", type=str, required=True, help="model for pose estimation (.pt).")
    parser.add_argument("--keep_full_image", action="store_true", help="Keep original image and express keypoints in global image coordinates.")


    args = parser.parse_args()
    print("============================================")
    print("== Starting pseudo-annotation pipeline... ==")
    print("============================================\n")
    preannotator = PreAnnotator(
        image_folder=args.input_dir,
        output_folder=args.output_dir,
        player_detector=YOLODetector(model_path=args.detection),
        pose_estimator=TorchEstimator(model_path=args.pose),
        keep_full_image=args.keep_full_image
    )
    preannotator.process_all_images()
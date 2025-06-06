from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import cv2
import json
import os
import torchvision.transforms as transforms
from PIL import Image
import PoseEstimator
import PlayerDetector

class PreAnnotator():
    """
    Abstract base class for a pre-annotation pipeline.
    This ensures modularity so any detection/pose model can be plugged in.
    """

    def __init__(self, image_folder, output_folder, player_detector, pose_estimator):
        self.padding = 40 # padding to add when cropping images
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
        """Pipeline: Detect players -> Crop -> Estimate Pose -> Save Annotations"""

        image = cv2.imread(image_path)
        h, w, _= image.shape
        if image is None:
            print(f"Error loading {image_path}")
            return None
        
        players = self.player_detector.detect_players(image)  # [(x1, y1, x2, y2), ...]

        if players is None:
            print(f"No person detected in {image_path}")
            return None
        
        annotations = []
        filename, file_extension = os.path.splitext(image_path)
        for i, (x1, y1, x2, y2) in enumerate(players):
            image_name = os.path.join(self.image_output_folder, f"{os.path.basename(filename)}_{i}{file_extension}" )
            cropped_player = image[max(0,y1-self.padding):min(h,y2+self.padding), max(0,x1-self.padding):min(w,x2+self.padding)]
            tensor = transforms.ToTensor()(Image.fromarray(cropped_player))
            preds = self.pose_estimator.estimate_pose(tensor)
            preds = preds[:].flatten()
            keypoints = []

            for k in range(0, len(preds), 2): # add the visibility parameter
                keypoints.append(preds[k])
                keypoints.append(preds[k+1])
                if (preds[k] > 1e-4 and preds[k+1] >1e-4) :
                    keypoints.append(2)
                else:
                    keypoints.append(0)

            annotations.append({
                "bbox": [x1, y1, x2-x1, y2-y1],
                "keypoints": keypoints,
                "image_name": f"{os.path.basename(filename)}_{i}{file_extension}",
                "image_shape": cropped_player.shape
            })
            cv2.imwrite(image_name, cropped_player)

        return annotations

    def process_all_images(self):
        """Processes all images in the folder and saves COCO GT annotations using kp_to_json."""
        image_metadata = []
        annotations = []
        image_id_map = {}  
        annotation_id = 1  
        print("Start of image processing... ")
        for filename in tqdm(os.listdir(self.image_folder), desc="Detection and pose estimation: "):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(self.image_folder, filename)
                annotation_data = self.process_image(image_path)

                if annotation_data:
                    for ann in annotation_data:
                        cropped_image_name = ann["image_name"]
                        width, height,_ = ann["image_shape"]

                        if cropped_image_name not in image_id_map:
                            image_id = len(image_id_map) + 1
                            image_id_map[cropped_image_name] = image_id

                        image_metadata.append({
                            "id": image_id,
                            "file_name": cropped_image_name,
                            "width": width,
                            "height": height,
                            "license": 0,
                            "flickr_url": "",
                            "coco_url": "",
                            "date_captured": 0
                        })

                        bbox = ann["bbox"]
                        keypoints = ann["keypoints"]

                        annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,  # Person class
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],  # width * height
                            "iscrowd": 0,
                            "num_keypoints": sum(1 for v in keypoints[2::3] if v > 0),
                            "keypoints": keypoints
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

    args = parser.parse_args()
    print("============================================")
    print("== Starting pseudo-annotation pipeline... ==")
    print("============================================\n")
    preannotator = PreAnnotator(
        image_folder=args.input_dir,
        output_folder=args.output_dir,
        player_detector=PlayerDetector.YOLODetector(model_path=args.detection),
        pose_estimator=PoseEstimator.TorchEstimator(model_path=args.pose)
    )
    preannotator.process_all_images()
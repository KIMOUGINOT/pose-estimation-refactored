import os
from time import time
import cv2
import argparse
from ultralytics import YOLO
from tqdm import tqdm

def detect_and_crop_players(input_dir, output_dir, model_path):
    """
    Detect players with Yolo and crop them in singular images.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
        model_path (str): Path to Yolo model.
    """
    start_time = time()
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    padding = 10 # to get the padel racket and for supposed better performance

    with tqdm(total=len(image_files), desc="Extracting players... ") as pbar:
        for img_name in image_files:
            img_path = os.path.join(input_dir, img_name)
            image = cv2.imread(img_path)

            h, w, _ = image.shape 

            if image is None:
                print(f"Error reading the image : {img_path}")
                continue
            
            results = model(img_path, verbose=False)

            for idx, result in enumerate(results):
                for det in result.boxes.xyxy:  # Format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, det[:4])  

                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)

                    cropped_player = image[y1:y2, x1:x2]

                    save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_player{idx}.jpg")
                    cv2.imwrite(save_path, cropped_player)

            pbar.update(1)
    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Work time : {elapsed_time:.3f} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and extract players.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input images directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output images directory.")
    parser.add_argument("--model", type=str, required=True, help="Yolo model for detection (.pt).")

    args = parser.parse_args()

    detect_and_crop_players(args.input_dir, args.output_dir, args.model)
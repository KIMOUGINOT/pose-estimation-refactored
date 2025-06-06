import os
import argparse
from tqdm import tqdm

def rename_images_and_labels(image_dir, label_dir, image_prefix):
    """Rename images and corresponding labels of a directory according to the name given in parameters

    Args:
        image_dir (str): Path of the images directory.
        label_dir (str): Path of the label directory.
        image_prefix (str): New root name to give to the images and the labels.
    """
    image_extensions = {'.jpg', '.jpeg', '.png'} 
    
    images = sorted([f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions])
    
    print(f"Renaming images from {image_dir} and labels from {label_dir} into {image_prefix}_x.yyy...")

    for idx, image_name in tqdm(enumerate(images)):
        old_image_path = os.path.join(image_dir, image_name)
        new_image_name = f"{image_prefix}_{idx+1}{os.path.splitext(image_name)[1].lower()}"
        new_image_path = os.path.join(image_dir, new_image_name)
        
        os.rename(old_image_path, new_image_path)
        
        old_label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")
        new_label_path = os.path.join(label_dir, os.path.splitext(new_image_name)[0] + ".txt")
        
        if os.path.exists(old_label_path):
            os.rename(old_label_path, new_label_path)

    print("Job done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename images and corresponding labels of a directory according to the name given in parameters.")
    parser.add_argument("--images_dir", type=str, required=True, help="Path of the images directory")
    parser.add_argument("--labels_dir", type=str, required=True, help="Path of the label directory")
    parser.add_argument("--prefix", type=str, required=True, help="New root name for renaming")
    
    args = parser.parse_args()
    
    rename_images_and_labels(
        image_dir=args.images_dir,
        label_dir=args.labels_dir,
        image_prefix=args.prefix
    )

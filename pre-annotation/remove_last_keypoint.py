import os
import argparse

def remove_last_keypoint(labels_dir):
    """
    Recursively remove the last keypoint from all YOLO label files (.txt) 
    in the given directory and its subdirectories.

    Args:
        labels_dir (str): Path to the dataset labels directory.
    """
    if not os.path.exists(labels_dir):
        print(f"Error: Directory '{labels_dir}' does not exist.")
        return

    modified_files = 0  # Counter for processed files

    for root, _, files in os.walk(labels_dir):
        for filename in files:
            if filename.endswith(".txt"):  # Process only .txt files
                file_path = os.path.join(root, filename)

                with open(file_path, 'r') as file:
                    lines = file.readlines()

                with open(file_path, 'w') as file:
                    for line in lines:
                        values = line.strip().split()
                        if len(values) >= 3:  # Ensure there are enough values to remove a keypoint
                            new_line = ' '.join(values[:-3])  # Remove the last 3 numbers
                            file.write(new_line + '\n')

                modified_files += 1

    print(f"Finished processing {modified_files} label files in: {labels_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove the last keypoint from all YOLO label files in a dataset.")
    parser.add_argument("--labels_dir", type=str, required=True, 
                        help="Path to the dataset labels directory (e.g., 'yolo_dataset/labels/')")

    args = parser.parse_args()

    remove_last_keypoint(args.labels_dir)

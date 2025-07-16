# Pose Estimation pipeline

This project focuses on estimating the pose of human (keypoint detection) using state of the art models such as ResNet, MobileNet or other backbones using timm's library. It contains tools for training, validation, model export, model benchmarking, tools for dataset making or even pre-annotation for CVAT.

## Project Structure

The repository is organized as follows:

```
.
├── config/
│
├── datasets/                       # Dataset files, metrics
│
├── pre-annotation/                 # Contains all the information to annotate a dataset
│
├── models/                         # Contains lightning data module, pose estimation module
│   ├── pose_datamodule.py
│   ├── pose_estimation_model.py          
│   └── collection/                 # Contains the models implementation
│
├── pre-annotation/                   
│   ├── PlayerDetector.py           # Player detection worker
│   ├── PoseEstimator.py.py         # Pose estimation worker
│   ├── PreAnnotator.py             # Pre annotate a whole image folder
│   └── ...
│
├── utils/                   
│   ├── benchmark.py                # Benchmark a model inference time
│   ├── export.py                   # Export a model to Torchscript for deployment
│   ├── visualizer.py               # Visualize dataset items
│   └── ...
│
├── README.md                       # Main project documentation
│
└── requirements.txt                # Python dependencies
```

## Prerequisites

- **Dependencies**: Install required packages using:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

### 1. Annotation pipeline
#### Prerequisite
Make sure to have all the images you want to process in a single folder. 
You can use the choose_frame.py script if you want to read a video and save specifics images ( blurry frame, occluded people, etc) for your future dataset.
```bash
python -m pre-annotation.choose_frame [PATH/TO/VIDEO]
```
You can also use the video_to_image_folder.py script if you want to save image clips of a video with steps ,etc (check the helper because this is very modular).
```bash
python -m pre-annotation.video_to_image_folder --video [PATH/TO/VIDEO] --output_dir [IMAGE_OUTPUT_FOLDER] --step [int] --clip_size [int] --clip_step [int] --start_idx [int] --max_idx [int] 
```
#### Pipeline running
Run the pipeline using the models of your choice both for people detection and pose estimation:
```bash
python -m pre-annotation.PreAnnotator --input_dir [IMAGE_FOLDER] --output_dir [IMAGE_OUTPUT_FOLDER] --detection [PLAYER_DETECTION_MODEL_PATH] --pose [POSE_ESTIMATION_MODEL_PATH] [--keep_full_image]
```
You can choose to keep the full image or the people's crops with the optional arg --keep_full_image.
Only yolo models work for detection as I've only implemented for YOLO. Also the pose model must have been exported to TorchScript.
##### Example
```bash
python -m pre-annotation.PreAnnotator --input_dir image_to_annotate/ --output_dir TO_CVAT/ --detection Player-Detection-YOLOv11X-2024-12.pt --pose xpose_resnet18.pt
```
#### Format
The folder created is a COCO type dataset compatible with the dataset importation of CVAT. Zip it and drag it on CVAT to load annotations.
The dataset will be organized as follows:

```
.
├── images/              
│   ├── image_1.jpg    
│   ├── image_2.jpg    
│   └── ...      
└── annotations/    
    └── person_keypoints_default.json                     
```
See the [CVAT Pose Annotation Guide](./pre-annotation/README.md) for following info.

### 2. Training Pipeline
#### Model Training
Train models using the model config file of your choice:
```bash
python -m train --cfg [MODEL_CONFIG_FILE].yaml
```
Output of the training such as training logs and model weights will be saved in `outputs/pose/` with vizualisation on tensorboard. Use `tensorboard --logdir outputs/pose/` to access the training monitor board. 

#### Model Evaluation
Evaluate the trained model's performance:
```bash
python -m test --cfg [MODEL_CONFIG_FILE].yaml
```

#### Model exportation to TorchScript
Export the model to TorchScript format:
```bash
python -m utils.export --cfg [MODEL_CONFIG_FILE].yaml
```
Use parser helper for more information.

#### Model inference time evaluation
Benchmark your TorchScript model:
```bash
python -m utils.benchmark --model [MODEL_PATH]
```
Use parser helper for more detailed args (batch size, image size, etc).

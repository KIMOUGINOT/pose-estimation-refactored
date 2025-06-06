# Pose Estimation pipeline

This project focuses on estimating the pose of human (keypoint detection) using state of the art models such as ResNet, MobileNet or other backbones using timm's library. It contains tools for training, validation, model export, model benchmarking, tools for dataset making or even pre-annotation for CVAT.

## Project Structure

The repository is organized as follows:

```
.
├── config/
├── datasets/              # Dataset files and utils
├── pre-annotation/        # Contains all the information to annotate a dataset
├── models/                # Contains lightning data module, pose estimation module
│   └── collection/        # Contains the models implementation
├── utils/                   
│   ├── benchmark.py       # Benchmark a model inference time
│   ├── export.py          # Export a model to Torchscript for deployment
│   ├── visualizer.py      # Visualize dataset items
│   └── ...
├── README.md              # Main project documentation
└── requirements.txt       # Python dependencies
```

## Prerequisites

- **Dependencies**: Install required packages using:
  ```bash
  pip install -r requirements.txt
  ```
- **Fetch models**: Access to a model zoo here: https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC.

## Usage

### 1. Annotation pipeline
#### Prerequisite
Make sure to have all the images you want to process in a single folder. You can also use the choose_frame.py script if you want to read a video and save specifics images (smash, blurry frame, occluded players, etc) for your future dataset.
```bash
python pre-annotation/choose_frame.py [PATH/TO/VIDEO]
```
#### Pipeline running
Run the pipeline using the models of your choice both for people detection and pose estimation:
```bash
python pre-annotation/PreAnnotator.py --input_dir [IMAGE_FOLDER] --output_dir [IMAGE_OUTPUT_FOLDER] --detection [PLAYER_DETECTION_MODEL_PATH] --pose [POSE_ESTIMATION_MODEL_PATH]
```
Only yolo models work for detection as I've only implemented for YOLO. Also the pose model must have been exported to TorchScript.
##### Example
```bash
python pre-annotation/PreAnnotator.py --input_dir image_to_annotate/ --output_dir TO_CVAT/ --detection Player-Detection-YOLOv11X-2024-12.pt --pose xpose_resnet18.pt
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
### 2. Training Pipeline
#### Model Training
Train models using the model config file of your choice:
```bash
python train.py --cfg [MODEL_CONFIG_FILE].yaml
```
Output of the training such as training logs and model weights will be saved in `outputs/pose/` with vizualisation on tensorboard. Use `tensorboard --logdir outputs/pose/` to access the training monitor board. 

#### Model Evaluation
Evaluate the trained model's performance:
```bash
python test.py --cfg [MODEL_CONFIG_FILE].yaml
```

#### Model exportation to TorchScript
Export the model to TorchScript format:
```bash
python utils/export.py --cfg [MODEL_CONFIG_FILE].yaml --ext [MODEL_EXTENSION]
```
Use parser helper for more information.

#### Model inference time evaluation
Benchmark your TorchScript model:
```bash
python tools/benchmark.py --model [MODEL_PATH]
```
Use parser helper for more detailed args (batch size, image size, etc).
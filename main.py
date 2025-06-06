from omegaconf import OmegaConf
from models.pose_datamodule import PoseDataModule
from utils import visualize_sample 
import random

cfg = OmegaConf.load("config/resnet.yaml")
datamodule = PoseDataModule(cfg.data)
datamodule.setup()

dataset = datamodule.train_dataset
# print(f"Dataset size: {len(dataset)}")

# i = random.randint(0, len(dataset)-1)
# sample = dataset[i]
# visualize_sample(sample, show_keypoints=True, show_bbox=True, save_path="visualize.jpg")

for j in range(505,509):
    sample = dataset[j]
    visualize_sample(sample, show_keypoints=True, show_bbox=True, save_path=f"visualize_{j}.jpg")
    

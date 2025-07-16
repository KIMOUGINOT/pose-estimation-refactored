import os
import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from models.pose_estimation_model import PoseEstimationModel
from models.pose_datamodule import PoseDataModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="config/resnet.yaml",
        help="Path to the config file (YAML)"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    logger = TensorBoardLogger(save_dir=cfg.train.output_dir, name="pose")
    print("===================================")
    print("== Starting testing pipeline... ==")
    print("===================================")

    datamodule = PoseDataModule(cfg.data)
    if "ckpt" in cfg.model.pretrained:
        model = PoseEstimationModel.load_from_checkpoint(checkpoint_path=cfg.model.pretrained, cfg=cfg)
    else:
        model = PoseEstimationModel(cfg)

    # Determine accelerator
    if torch.cuda.is_available():
        accelerator = "cuda"
        devices = "auto"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        default_root_dir=cfg.train.output_dir
    )

    trainer.validate(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
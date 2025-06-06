import os
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from models.pose_estimation_model import PoseEstimationModel
from models.pose_datamodule import PoseDataModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/resnet.yaml",
        help="Path to the config file (YAML)"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    logger = TensorBoardLogger(save_dir=cfg.train.output_dir, name="pose")
    print("===================================")
    print("== Starting testing pipeline... ==")
    print("===================================")

    datamodule = PoseDataModule(cfg.data)
    if "ckpt" in cfg.model.pretrained:
        model = PoseEstimationModel.load_from_checkpoint(checkpoint_path=cfg.model.pretrained, cfg=cfg)
    else:
        model = PoseEstimationModel(cfg)

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=cfg.train.output_dir
    )

    trainer.validate(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
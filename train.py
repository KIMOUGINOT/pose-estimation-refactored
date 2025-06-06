import os
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import torch

from models.pose_estimation_model import PoseEstimationModel
from models.pose_datamodule import PoseDataModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the config file (YAML)"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    logger = TensorBoardLogger(save_dir=cfg.train.output_dir, name="pose")
    print("===================================")
    print("== Starting training pipeline... ==")
    print("===================================")

    datamodule = PoseDataModule(cfg.data)

    if "ckpt" in cfg.model.pretrained:
        model = PoseEstimationModel.load_from_checkpoint(checkpoint_path=cfg.model.pretrained, cfg=cfg)
        print(f"Using pretrained model from {cfg.model.pretrained}.")
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
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)], 
        default_root_dir=cfg.train.output_dir
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()

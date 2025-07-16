import os
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from omegaconf import OmegaConf
import torch

from ia.pose_estimation_2D.models.pose_estimation_model import PoseEstimationModel
from ia.pose_estimation_2D.models.pose_datamodule import PoseDataModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="config/default.yaml",
        help="Path to the config file (YAML)"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
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

    monitor = 'metrics/group_1/AP@50:95' #'val_loss'
    mode = 'max' # min
    checkpoint_callback = ModelCheckpoint(
        monitor= monitor, 
        mode=mode,
        save_top_k = 1,
        dirpath='outputs/pose'
        )

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback], #EarlyStopping(monitor=monitor, mode=mode, patience=40)
        default_root_dir=cfg.train.output_dir
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()

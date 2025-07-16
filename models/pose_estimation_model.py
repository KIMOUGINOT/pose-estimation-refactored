import torch
import torch.nn as nn
import pytorch_lightning as pl
import cv2
import numpy as np
from PIL import Image
import io
import torchvision.transforms.functional as F

from models import collection
import datasets as datasets

class PoseEstimationModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()  # pour loguer la config
        self.cfg = cfg
        self.evaluator = None 
        self._debug_images = [] ########## DEBUG ONLY ##########
        self._debug_preds = [] ########## DEBUG ONLY ##########
        self._debug_gts = [] ########## DEBUG ONLY ##########


        self.metric = datasets.CustomKeypointMetric(config_key=self.cfg.train.evaluation_config)
        
        self.model = eval("collection."+cfg.model.name)(cfg.model)
        if ".pt" in cfg.model.pretrained :
            self.model.load_state_dict(torch.load(cfg.model.pretrained), strict=False)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets, target_weight, _ = self._get_inputs_and_targets(batch)
        preds = self(images)

        if preds.shape[1] > targets.shape[1]:
            preds = preds[:, :targets.shape[1]]

        loss = self.criterion(preds * target_weight, targets * target_weight)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, target_weight, _ = self._get_inputs_and_targets(batch)
        preds = self(images)

        if preds.shape[1] > targets.shape[1]:
            preds = preds[:, :targets.shape[1]]

        loss = self.criterion(preds * target_weight, targets * target_weight)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._store_debug_data(batch, images, targets, preds)

        self.metric.update_from_batch(preds, batch)

        return loss

    def on_validation_epoch_end(self):
        summary = self.metric.compute()
        for group, results in summary.items():
            for k, v in results.items():
                if 'AP' in k:
                    self.log(f"metrics/{group}/{k}", v, prog_bar=False)
        # self.metric.log_to_tensorboard(self.logger, self.current_epoch)
        self.metric.summarize()
        self.metric.reset()

        if hasattr(self.logger, "experiment") and len(self._debug_images) > 0:
            image = self._debug_images[0]
            heatmap = self._debug_preds[0]
            joints_gt = self._debug_gts[0]

            coords_pred = []
            for k in range(min(heatmap.shape[0], 18)):
                hm = heatmap[k]
                y, x = np.unravel_index(hm.argmax(), hm.shape)
                conf = float(hm[y, x])
                coords_pred.append([x, y])

            coords_pred = np.array(coords_pred)
            coords_pred = datasets.adjust_preds_to_shape(torch.tensor(coords_pred[None, ...]), heatmap.shape[1:], (256,192)).numpy()[0]
            vis = self._draw_keypoints_pred_gt_on_tensor(image, coords_pred, joints_gt)
            self.logger.experiment.add_image("val/debug_pred_vs_gt", vis, self.current_epoch)
            
        # Reset
        self._debug_preds.clear()
        self._debug_images.clear()
        self._debug_gts.clear()
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        images, heatmaps = self._get_inputs_and_targets(batch)
        preds = self(images)
        loss = self.criterion(preds, heatmaps)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.train.max_epochs,  
            eta_min=self.cfg.train.min_lr     
            )

        return [optimizer], [scheduler]


    def _get_inputs_and_targets(self, batch):
        """
        Extracts image tensor, target heatmap, target_weight, and joints info from a batched dictionary.
        """
        if isinstance(batch, dict):
            image_tensor = batch["image"]
            target = batch["target"]
            target_weight = batch["target_weight"].unsqueeze(-1)  # shape: [B, K, 1, 1]
            joints_info = {
                "joints": batch["joints"],
                "joints_vis": batch["joints_vis"]
            }
            return image_tensor, target, target_weight, joints_info
        else:
            raise ValueError("Unsupported batch format")

    def _store_debug_data(self, batch, images, heatmaps, preds):
        self._debug_images.append(images[0].detach().cpu())
        self._debug_preds.append(preds[0].detach().cpu())

        joints = batch["joints"][0].detach().cpu()
        joints_vis = batch["joints_vis"][0].detach().cpu()
        self._debug_gts.append(torch.cat([joints, joints_vis], dim=1))

    def _draw_keypoints_on_tensor(self, image_tensor, keypoints, color=(255, 0, 0)):
        """
        Dessine les keypoints sur une image normalisée [3, H, W] (Tensor),
        les renvoie sous forme de tensor [3, H, W] (float entre 0 et 1) pour TensorBoard.
        """
        def denormalize_tensor(tensor, mean, std):
            mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
            std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
            return tensor * std + mean

        # Denormalize the image
        image = denormalize_tensor(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        image = (image * 255).clip(0, 255).astype(np.uint8).copy()

        for x, y, conf in keypoints:
            if conf > 0.0001:
                cv2.circle(image, (int(x), int(y)), radius=3, color=color, thickness=-1)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 #for tensorboard
        return image_tensor

    def _draw_keypoints_pred_gt_on_tensor(self, image_tensor, pred_kpts, gt_kpts, color_pred=(255, 0, 0), color_gt=(0, 255, 0)):
        """
        Affiche à la fois les keypoints prédits et ground truth sur une image [3, H, W] normalisée.
        - pred_kpts: [(x, y, conf)]
        - gt_kpts:   [(x, y, vis)]  ou torch.Tensor [K, 3]
        """
        def denormalize_tensor(tensor, mean, std):
            mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
            std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
            return tensor * std + mean

        # Denormalize the image
        image = denormalize_tensor(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        image = (image * 255).clip(0, 255).astype(np.uint8).copy()

        for x, y in pred_kpts:
            cv2.circle(image, (int(x), int(y)), 3, color_pred, -1)

        for x, y, v in gt_kpts:
            if v > 0 :
                cv2.circle(image, (int(x), int(y)), 3, color_gt, -1)

        # For tensorboard
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
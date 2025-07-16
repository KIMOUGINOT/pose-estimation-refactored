import torch.nn as nn
from .head_registry import HEADS
import timm


class ModularNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.validate_config(cfg)

        backbone_name = cfg.backbone
        head_name = cfg.head
        pretrained = cfg.is_backbone_pretrained

        self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                )

        feature_dims = self.backbone.feature_info.channels()

        if head_name not in HEADS:
            raise ValueError(f"Unknown head: {head_name}")

        self.head = HEADS[head_name](
            num_keypoints=cfg.num_keypoints,
            output_size=tuple(cfg.heatmap_size),
            image_size=tuple(cfg.image_size),
            feature_dim=feature_dims
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def validate_config(self, cfg):
        required_keys = [
            ("backbone"),
            ("head"),
            ("is_backbone_pretrained"),
            ("num_keypoints"),
            ("heatmap_size"),
            ("image_size"),
        ]

        for key in required_keys:

            if not hasattr(cfg, key):
                raise ValueError(f"Missing key '{key}' in config.")

        # Validation de types & formats
        if not isinstance(cfg.num_keypoints, int) or cfg.num_keypoints <= 0:
            raise ValueError("cfg.num_keypoints must be a positive integer.")

        if not (len(cfg.heatmap_size) == 2):
            raise ValueError("cfg.heatmap_size must be a tuple/list of 2 integers.")

        if not (len(cfg.image_size) == 2):
            raise ValueError("cfg.image_size must be a tuple/list of 2 integers.")




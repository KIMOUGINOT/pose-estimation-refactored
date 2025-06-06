from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import math
import cv2
import json
import os
from time import time
import _init_paths
import models

class TorchEstimator():
    def __init__(self, model_path, device=None):
        """
        Initializes the pose estimation model.

        Args:
            model_path (str): TorchScript pose estimation model.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")  
        self.model = torch.jit.load(model_path)
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((256, 192)),
        ])
        print(f"¤ Using Torch model from {model_path} for 2D pose estimation.")    

    def estimate_pose(self, input_tensor):
        """
        Returns:
            list: Flattened list of keypoints.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        image_shape = input_tensor.shape[2:]
        input_tensor = self.transform(input_tensor)

        with torch.no_grad():
            output = self.model(input_tensor)
            keypoints, _ = get_final_preds(output.clone().cpu().numpy(), image_shape)  # input_tensor shape: (1, 3, H, W)

        return keypoints
    
def get_final_preds(batch_heatmaps, image_shape):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    image_height = image_shape[0]
    image_width = image_shape[1]
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array(
                    [
                        hm[py][px+1] - hm[py][px-1],
                        hm[py+1][px]-hm[py-1][px]
                    ]
                )
                coords[n][p] += np.sign(diff) * .25

    preds = coords[0].copy()
    ones = np.ones((batch_heatmaps.shape[1], 1))
    preds_hom = np.concatenate([preds, ones], axis=1)
    scale_x = image_width / heatmap_width
    scale_y = image_height / heatmap_height
    A = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0]])

    transformed_coords = preds_hom @ A.T

    return transformed_coords, maxvals
    
def get_max_preds(batch_heatmaps):
    '''
    Extracts keypoint coordinates and confidence scores from a batch of heatmaps.
    
    Args:
        batch_heatmaps (np.ndarray): shape (batch_size, num_keypoints, height, width)
    
    Returns:
        preds (np.ndarray): shape (batch_size, num_keypoints, 2), coordinates (x, y)
        max_values (np.ndarray): shape (batch_size, num_keypoints, 1), confidence scores
    '''
    assert isinstance(batch_heatmaps, np.ndarray), 'Input must be a numpy ndarray'
    assert batch_heatmaps.ndim == 4, 'Input must be 4-dimensional (B, K, H, W)'

    batch_size, num_keypoints, heatmap_height, heatmap_width = batch_heatmaps.shape

    flat_heatmaps = batch_heatmaps.reshape((batch_size, num_keypoints, -1))

    max_indices = np.argmax(flat_heatmaps, axis=2)
    max_values = np.max(flat_heatmaps, axis=2)

    max_values = max_values.reshape((batch_size, num_keypoints, 1))
    max_indices = max_indices.reshape((batch_size, num_keypoints, 1))

    keypoint_coords = np.tile(max_indices, (1, 1, 2)).astype(np.float32)
    keypoint_coords[:, :, 0] = keypoint_coords[:, :, 0] % heatmap_width  
    keypoint_coords[:, :, 1] = np.floor(keypoint_coords[:, :, 1] / heatmap_width) 

    visibility_mask = np.tile((max_values > 0.0), (1, 1, 2)).astype(np.float32)
    keypoint_coords *= visibility_mask

    return keypoint_coords, max_values

if __name__ == "__main__" :

    estimateur = TorchEstimator("models/pose_coco/xpose_mobilenetv2_256x192_60_epochs.pth")
    input_tensor = torch.randn(10, 3, 256, 192)

    start = time.time()
    output = estimateur.model(input_tensor)
    end = time.time()
    print(f"inference time: {end-start} sec")
from abc import ABC, abstractmethod
from tqdm import tqdm
from ultralytics import YOLO
import torch
import cv2
import json
import os

class PlayerDetector(ABC):
    """Abstract class for player detection models."""
    
    @abstractmethod
    def detect_players(self, image):
        """Detect players in an image and return bounding boxes."""
        pass



class YOLODetector(PlayerDetector):
    def __init__(self, model_path, device=None):
        """
        Initializes YOLO-based player detection model.

        Args:
            model_path (str): Path to the YOLO model file (.pt).
            device (str, optional): 'cuda', 'cpu', or 'mps'. If None, auto-detects best available.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)  # Load model on correct device
        print(f"¤ Using model Yolo from {model_path} for people detection.")


    def detect_players(self, image, conf_threshold=0.6):
        """
        Returns:
            list of tuples: [(x1, y1, x2, y2), ...] for detected players.
        """
        results = self.model(image, device=self.device, verbose=False)[0]  # Run inference on the correct device
        players = [
            tuple(map(int, det)) for det, conf in zip(results.boxes.xyxy, results.boxes.conf) if conf >= conf_threshold
        ]

        return players

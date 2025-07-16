import albumentations as A
from albumentations.pytorch import ToTensorV2
import random as rd
import numpy as np


def get_pose_transforms(image_size, is_train=False, p_flip=0):
    w, h = image_size
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_train:
        return A.Compose([
            A.Rotate(limit=40, p=0.8),
            A.RandomScale(scale_limit=0.2, p=1.0),
            A.Resize(height=h, width=w),
            A.HorizontalFlip(p=p_flip),
            A.OneOf([
                A.ToGray(p=1.0),
                A.ChannelDropout(p=1.0)
            ], p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        return A.Compose([
            A.Resize(height=h, width=w),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def apply_flip_joints(kp: np.ndarray, flip_pairs):
    """
    Permute les articulations gauche/droite.
    """
    flipped = kp.copy()

    for i, j in flip_pairs:
        flipped[i], flipped[j] = flipped[j].copy(), flipped[i].copy()

    return flipped


def transform(image, kp2d, image_size, is_train=False, kp3d=None):
    """
    Applique les augmentations à l'image et aux keypoints 2D.
    Si kp3d est fourni, le modifie aussi en cas de flip.
    
    Args:
        image (np.ndarray): image d'entrée
        kp2d (np.ndarray): (N, 2)
        image_size (tuple): (width, height)
        is_train (bool): 
        kp3d (np.ndarray, optional): (N, 3) - keypoints 3D caméra
        
    Returns:
        image (Tensor)
        kp2d (np.ndarray)
        kp3d (np.ndarray or None)
        do_flip (bool)
    """
    do_flip = False
    p = 0
    if is_train:
        do_flip = rd.random() > 0.5
        p = 1 if do_flip else 0

    transform_pipeline = get_pose_transforms(image_size=image_size, is_train=is_train, p_flip=p)

    transformed = transform_pipeline(image=image, keypoints=kp2d)
    image = transformed["image"]
    kp2d = np.array(transformed["keypoints"])

    if do_flip:
        flip_pairs = [(i, i+1) for i in range(1,len(kp2d)-1,2)] #-1 a tester sur la 3D
        kp2d = apply_flip_joints(kp2d, flip_pairs)
        if kp3d is not None:
            kp3d[:, 0] *= -1
            kp3d = apply_flip_joints(kp3d, flip_pairs)

    return image, kp2d, kp3d 

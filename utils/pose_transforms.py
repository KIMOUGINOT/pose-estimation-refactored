import albumentations as A
from albumentations.pytorch import ToTensorV2
import random as rd

def get_pose_transforms(image_size, is_train=True):
    w, h = image_size
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    do_flip = False

    if is_train:
        do_flip = rd.random() > 0.5
        p = 1 if do_flip else 0

        return A.Compose([
            A.Rotate(limit=40, p=0.8),
            A.RandomScale(scale_limit=0.3, p=1.0),
            A.Resize(height=h, width=w),
            A.HorizontalFlip(p=p),
            A.OneOf([A.ToGray(p=1.0),
                     A.ChannelDropout(p=1.0)], 
                     p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), do_flip
    else:
        return A.Compose([
            A.Resize(height=h, width=w),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), do_flip 
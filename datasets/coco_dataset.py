import torch
from torch.utils.data import Dataset
from datasets.coco_annotation_loader import AnnotationLoader
from datasets.sample_builder import SampleBuilder
from utils.pose_transforms import transform 
from PIL import Image
import numpy as np
import os
import pickle
from tqdm import tqdm
import cv2


class COCODataset(Dataset):
    def __init__(self, annotation_file, image_root, image_size=(192, 256), num_keypoints=17, is_train=False, target_generator=None, use_gt_bbox=None):
        self.image_root = image_root
        self.target_generator = target_generator
        self.is_train = is_train
        self.use_gt_bbox = use_gt_bbox
        self.num_keypoints = num_keypoints
        self.image_size = image_size
        self.pairs_to_flip = [(i,i+1) for i in range(1,17,2)]

        self.loader = AnnotationLoader(annotation_file)
        self.builder = SampleBuilder(
            num_keypoints=num_keypoints,
            image_size=image_size,
            aspect_ratio=image_size[0] / image_size[1],
            image_root=self.image_root,
            use_gt_bbox = self.use_gt_bbox
        )

        self.db = self._build_db()

    def _build_db(self):
        """
        """
        print("[INFO] Building dataset DB...")
        samples = []

        for img_id in tqdm(self.loader.get_image_ids(), desc="Building Samples"):
            img_info = self.loader.get_image_info(img_id)
            anns = self.loader.get_annotations_for_image(img_id)
            sample = self.builder.build_sample_metadata(img_info, anns, img_id)
            if sample is not None:
                if isinstance(sample, list):
                    samples += sample
                else :
                    samples.append(sample)

        print(f"[INFO] Done. Total valid samples: {len(samples)}")
        return np.array(samples)


    def __getitem__(self, index):
        sample = self.db[index]

        image = cv2.imread(sample['image_path'])
        if image is None:
            raise FileNotFoundError(f"Image not found: {sample['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        joints = sample['joints']
        keypoints = joints.tolist()

        if sample['bbox'] != [0, 0, 0, 0]: #keep the global image
            trans = get_affine_transform(sample['center'], sample['scale'], self.image_size)
            image = cv2.warpAffine(image, trans, self.image_size)
            keypoints = [affine_transform(kpt, trans) for kpt in keypoints]

        image, keypoints, _ = transform(image, keypoints, self.image_size, is_train=self.is_train)

        joints = torch.tensor(keypoints, dtype=torch.float32)

        target = self.target_generator(joints, sample['joints_vis'])
        target_weight = torch.tensor(sample['joints_vis'], dtype=torch.float32).reshape(-1, 1)

        return  {
            'image': image,
            'target': target,
            'target_weight': target_weight,
            'joints': joints,
            'joints_vis': torch.tensor(sample['joints_vis'], dtype=torch.float32).reshape(-1, 1),
            'kp3d': torch.zeros((joints.shape[0], 3), dtype=torch.float32) #for 3D implem        
            }

    def __len__(self):
        return len(self.db)


def get_affine_transform(center, scale, output_size):
    """Builds the affine transformation matrix."""

    if isinstance(scale, (tuple, list, np.ndarray)):
        scale = sum(scale) * 0.5

    src_w = scale * 200.0
    dst_w, dst_h = output_size

    src_dir = np.array([0, src_w * -0.5])
    dst_dir = np.array([0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = [center[0] - src_dir[1], center[1] + src_dir[0]]

    dst[0, :] = [dst_w / 2, dst_h / 2]
    dst[1, :] = dst[0, :] + dst_dir
    dst[2, :] = [dst[0, 0] - dst_dir[1], dst[0, 1] + dst_dir[0]]

    trans = cv2.getAffineTransform(src, dst)
    return trans


def affine_transform(pt, trans):
    """Applies affine transformation to a single keypoint."""
    new_pt = np.array([pt[0], pt[1], 1.])
    new_pt = np.dot(trans, new_pt)
    return new_pt[:2]
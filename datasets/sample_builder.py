import numpy as np
import cv2
import os

class SampleBuilder:
    def __init__(self, num_keypoints, image_size, image_root, aspect_ratio=1.0, pixel_std=200, use_gt_bbox=True):
        self.num_keypoints = num_keypoints
        self.image_width, self.image_height = image_size
        self.aspect_ratio = aspect_ratio
        self.pixel_std = pixel_std
        self.image_root = image_root
        self.use_gt_bbox = use_gt_bbox

    def build_sample_metadata(self, image_info, annotations, image_id):
        valid_objs = [ann for ann in annotations if ann.get('num_keypoints', 0) > 0]
        samples = []

        if not valid_objs:
            return None

        for ann in valid_objs:
            joints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            joints_xy = joints[:, :2]
            joints_vis = joints[:, 2]

            K = self.num_keypoints
            current_K = joints.shape[0]

            if current_K < K: #match keypoint length
                pad_joints = np.zeros((K, 2), dtype=np.float32)
                pad_vis = np.zeros((K,), dtype=np.float32)
                pad_joints[:current_K] = joints_xy
                pad_vis[:current_K] = joints_vis
                joints_xy = pad_joints
                joints_vis = pad_vis
            elif current_K > K:
                joints_xy = joints_xy[:K]
                joints_vis = joints_vis[:K]

            center, scale = self._box_to_center_scale(ann['bbox'])

            sample = {
                'image_id': image_id,
                'image_path': os.path.join(self.image_root, image_info['file_name']),
                'joints': joints_xy,
                'joints_vis': joints_vis,
                'center': center,
                'scale': scale,
                'bbox': ann['bbox'] if self.use_gt_bbox else [0, 0, 0, 0],
            }
            samples.append(sample)

        return samples

    def _sanitize_bbox(self, bbox, width, height):
        x, y, w, h = bbox
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width - 1, x1 + max(0, w - 1))
        y2 = min(height - 1, y1 + max(0, h - 1))
        return [x1, y1, x2 - x1, y2 - y1]

    def _box_to_center_scale(self, box):
        x, y, w, h = box
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array([w / self.pixel_std, h / self.pixel_std], dtype=np.float32)
        scale *= 1.25
        return center, scale

    def _get_affine_transform(self, center, scale, output_size):
        src_w = scale[0] * self.pixel_std
        dst_w, dst_h = output_size

        src_dir = np.array([0, src_w * -0.5], dtype=np.float32)
        dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)

        src[0, :] = center
        src[1, :] = center + src_dir
        src[2, :] = [center[0] - src_dir[1], center[1] + src_dir[0]]

        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = [dst_w * 0.5, dst_h * 0.5] + dst_dir
        dst[2, :] = [dst_w * 0.5 - dst_dir[1], dst_h * 0.5 + dst_dir[0]]

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans


from collections import OrderedDict
from collections import defaultdict
import torch
import json
import numpy as np
import cv2
import os

class CustomKeypointMetric:
    def __init__(self, config_key="COCO_and_racket", oks_thresholds=None):
        self.config_key = config_key
        self.config = {
            'COCO_and_racket': [[i for i in range(17)], [17]],
            'COCO': [[i for i in range(17)]],
            'racket': [[17]],
            'useful_keypoints': [[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]],
            'sigmas': [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,
                       .62, 1.07, 1.07, .87, .87, .89, .89, .89]  # don't forget to /10 after (done)
        }

        self.oks_thresholds = oks_thresholds or np.arange(0.5, 1.0, 0.05)
        self.predictions = []  # List of (pred, gt, vis)

    def update(self, preds, gts, visibilities):
        for pred, gt, vis in zip(preds, gts, visibilities):
            self.predictions.append((pred, gt, vis))

    def update_from_batch(self, preds, batch):
        joints = batch["joints"]  # (N, K, 2)
        joints_vis = batch["joints_vis"].squeeze(-1) # (N, K)

        pred_kpts = get_final_preds(preds)  # (N, K, 2)
        pred_kpts = adjust_preds_to_shape(pred_kpts, (64, 48), (256, 192))  

        pred_kpts_with_scores = torch.cat([
            pred_kpts,
            torch.ones_like(joints_vis.unsqueeze(-1))
        ], dim=2)

        self.update(
            preds=pred_kpts_with_scores.detach().cpu().numpy(),
            gts=joints.detach().cpu().numpy(),
            visibilities=joints_vis.detach().cpu().numpy()
        )

    def compute(self):
        self.check_config_key()
        grouped_results = {f"group_{i}": [] for i in range(len(self.config[self.config_key]))}
        ap_results = {f"group_{i}": [] for i in range(len(self.config[self.config_key]))}

        for pred_kpts, gt_kpts, vis in self.predictions:
            kpts_flat = pred_kpts[:, :2].flatten()
            gt_flat = gt_kpts.flatten()
            vis_flat = vis.flatten()

            for idx, keypoints_group in enumerate(self.config[self.config_key]):
                sigmas = [self.config['sigmas'][k] for k in keypoints_group]
                oks = self.computeOKS(
                    gt_flat, kpts_flat, sigmas, keypoints_group, vis_flat
                )
                grouped_results[f"group_{idx}"].append(oks)
                ap_results[f"group_{idx}"].append([
                    1 if oks > thr else 0 for thr in self.oks_thresholds
                ])

        summary = {}
        for key in grouped_results:
            ap_matrix = np.array(ap_results[key]) 

            ap = ap_matrix.mean(axis=0)
            ar = ap_matrix.sum(axis=0) / len(ap_matrix)

            summary[key] = {
                "AP@50": float(ap[self._get_thr_index(0.50)]),
                "AP@75": float(ap[self._get_thr_index(0.75)]),
                "AP@50:95": float(np.mean(ap)),
                "AR@50": float(ar[self._get_thr_index(0.50)]),
                "AR@75": float(ar[self._get_thr_index(0.75)]),
                "AR@50:95": float(np.mean(ar)),
            }

        return summary

    def _get_thr_index(self, value):
        idx = np.where(np.isclose(self.oks_thresholds, value))[0]
        if len(idx) == 0:
            raise ValueError(f"Seuil {value} non présent dans oks_thresholds")
        return idx[0]

    def computeOKS(self, g, d, sig, indices, visibility):
        if len(g) == 0 or len(d) == 0:
            return 0

        sigmas = np.array(sig) / 10.0
        vars = (sigmas * 2) ** 2

        xg = np.array([g[2 * k] for k in indices])
        yg = np.array([g[2 * k + 1] for k in indices])
        vg = np.array([visibility[k] for k in indices])

        xd = np.array([d[2 * k] for k in indices])
        yd = np.array([d[2 * k + 1] for k in indices])

        area = 192 * 256  # image size fixée

        k1 = np.count_nonzero(vg > 0)
        if k1 > 0:
            dx = xd - xg
            dy = yd - yg
            e = (dx**2 + dy**2) / vars / (area + np.spacing(1)) / 2
            e = e[vg > 0]
        else:
            z = np.zeros_like(xg)
            dx = np.maximum(z, xd) + np.maximum(z, 192 - xd)
            dy = np.maximum(z, yd) + np.maximum(z, 256 - yd)
            e = (dx**2 + dy**2) / vars / (area + np.spacing(1)) / 2

        oks = np.sum(np.exp(-e)) / len(e)
        return oks

    def reset(self):
        self.predictions = []

    def check_config_key(self):
        num_keypoint = self.predictions[0][0].shape[0]
        if num_keypoint < 18 and self.config_key in ['COCO_and_racket', 'racket']:
            print(f"[WARNING] Can't evaluate with {self.config_key} config if the number of keypoints is {num_keypoint}. Falling back to COCO config.")
            self.config_key = 'COCO'

    def log_to_tensorboard(self, logger, step):
        summary = self.compute()
        for group, results in summary.items():
            for k, v in results.items():
                if 'AP@50:95' in k:
                    logger.experiment.add_scalar(f"metrics/{group}/{k}", v, step)

    def summarize(self):
        import tabulate
        summary = self.compute()
        table = []
        for group, res in summary.items():
            row = [group] + [f"{res[k]:.4f}" for k in res]
            table.append(row)

        headers = [self.config_key] + list(next(iter(summary.values())).keys())
        print("\n")
        print(tabulate.tabulate(table, headers=headers, tablefmt="fancy_grid"))

def adjust_preds_to_shape(preds, input_shape, output_shapes):
    """
    Vectorized affine transform from heatmap coordinates to original crop coordinates.

    Args:
        preds: (B, K, 2)
        input_shape: (H_in, W_in)
        output_shapes: either (H_out, W_out) or list of (H_out, W_out)

    Returns:
        tensor (B, K, 2) - Keypoints coordinates in the global image referential.
    """
    B, K, _ = preds.shape
    H_in, W_in = input_shape

    # Si output_shapes est un seul tuple, on le transforme en liste
    if isinstance(output_shapes, tuple):
        output_shapes = [output_shapes] * B
    elif isinstance(output_shapes, list):
        if len(output_shapes) != B:
            raise ValueError(f"Expected output_shapes of length {B}, got {len(output_shapes)}")
    else:
        raise TypeError(f"output_shapes must be a tuple or list of tuples, got {type(output_shapes)}")

    transforms = []
    for out_h, out_w in output_shapes:
        scale_x = out_w / W_in
        scale_y = out_h / H_in
        A = torch.tensor([
            [scale_x, 0, 0],
            [0, scale_y, 0]
        ], dtype=preds.dtype, device=preds.device)
        transforms.append(A)

    transforms = torch.stack(transforms, dim=0)  # (B, 2, 3)

    ones = torch.ones((B, K, 1), dtype=preds.dtype, device=preds.device)
    preds_homo = torch.cat([preds, ones], dim=2)  # (B, K, 3)

    transformed = torch.bmm(preds_homo, transforms.transpose(1, 2))  # (B, K, 2)

    return transformed


def get_final_preds(batch_heatmaps):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n, p]
            px = int(torch.floor(coords[n, p, 0] + 0.5).item())
            py = int(torch.floor(coords[n, p, 1] + 0.5).item())
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = torch.tensor([
                    hm[py, px + 1] - hm[py, px - 1],
                    hm[py + 1, px] - hm[py - 1, px]
                ], device=coords.device)
                coords[n, p] += torch.sign(diff) * 0.25

    return coords

    
def get_max_preds(batch_heatmaps):
    B, K, H, W = batch_heatmaps.shape

    flat = batch_heatmaps.view(B, K, -1)
    maxvals, idx = torch.max(flat, dim=2)

    maxvals = maxvals.view(B, K, 1)
    idx = idx.view(B, K, 1).float()

    coords = torch.zeros((B, K, 2), device=batch_heatmaps.device)
    coords[:, :, 0] = idx[:, :, 0] % W  # x
    coords[:, :, 1] = torch.floor(idx[:, :, 0] / W)  # y

    mask = (maxvals > 0).float()
    coords *= mask

    return coords, maxvals
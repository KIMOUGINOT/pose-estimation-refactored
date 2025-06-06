import numpy as np
import torch


class HeatmapTargetGenerator:
    def __init__(self, output_size, sigma=0.1):
        self.output_size = output_size
        self.sigma = sigma

    def __call__(self, joints, joints_vis):
        return generate_target_heatmaps(joints, joints_vis, self.output_size, self.sigma)

def generate_target_heatmaps(joints, joints_vis, output_size, sigma=0.1):
    """
    Génère des heatmaps
    
    joints: (K, 2)
    joints_vis: (K,)
    output_size: (W, H)
    """
    K = joints.shape[0]
    H, W = output_size
    heatmaps = np.zeros((K, H, W), dtype=np.float32)

    tmp_size = sigma * 3

    for k in range(K):
        if joints_vis[k] == 0:
            continue

        mu_x = int(joints[k, 0] / 4 + 0.5)  #/4 pour arriver à la bonne shape de heatmap, à changer si on est pu sur des heatmaps de 64x48
        mu_y = int(joints[k, 1] / 4 + 0.5)

        if mu_x < 0 or mu_y < 0 or mu_x >= W or mu_y >= H:
            continue

        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
            continue

        size = int(2 * tmp_size + 1)
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
        img_x = max(0, ul[0]), min(br[0], W)
        img_y = max(0, ul[1]), min(br[1], H)

        heatmaps[k, img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return torch.from_numpy(heatmaps).float()


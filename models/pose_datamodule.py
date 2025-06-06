from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from datasets import COCODataset
from utils import HeatmapTargetGenerator, get_pose_transforms
import os


class PoseDataModule(LightningDataModule):
    def __init__(self, cfg_data):
        super().__init__()
        self.cfg = cfg_data
        self.target_generator = HeatmapTargetGenerator(output_size=(64, 48), sigma=2)

    def setup(self, stage=None):
        self.train_dataset = self._load_concat_dataset("train", train=True)

        if "val" in self.cfg.splits:
            self.val_dataset = self._load_concat_dataset("val", train=False)
        else:
            self.val_dataset = None

        if "test" in self.cfg.splits:
            self.test_dataset = self._load_concat_dataset("test", train=False)
        else:
            self.test_dataset = None

    def _load_concat_dataset(self, split="train", train=True):
        datasets = []
        for i,root in enumerate(self.cfg.dataset_roots):
            image_dir = os.path.join(root, "images",split)
            ann_file = os.path.join(root, "annotations", self.cfg.splits[split])

            if not os.path.isfile(ann_file):
                print(f"Missing annotation file: {ann_file}")
            else :
                dataset = COCODataset(
                    annotation_file=ann_file,
                    image_root=image_dir,
                    image_size=tuple(self.cfg.image_size),
                    num_keypoints=self.cfg.num_keypoints,
                    target_generator=self.target_generator,
                    transform=self._build_transforms(train=train),
                    use_gt_bbox=self.cfg.use_gt_bbox[i]
                )
                datasets.append(dataset)

        # if split is "val" :
        #     return datasets
        if len(datasets) == 1:
            return datasets[0]
        datasets = ConcatDataset(datasets)
        print(f"[INFO] Datasets concatenated. Total valid samples: {len(datasets)}")
        
        return datasets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )

    def _build_transforms(self, train):
        return get_pose_transforms(self.cfg.image_size, is_train=train)
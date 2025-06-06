from collections import OrderedDict
from datasets import CocoEvaluator
from torch.utils.data import ConcatDataset
import os

class MultiCocoEvaluator:
    def __init__(self, datasets, output_dir):
        self.datasets = datasets.datasets if isinstance(datasets, ConcatDataset) else [datasets]
        self.output_dir = output_dir

    def evaluate(self, all_preds, all_metas, epoch):
        results = {}
        idx = 0

        for i, dataset in enumerate(self.datasets):
            num_samples = len(dataset)
            coco_gt = dataset.loader.coco
            evaluator = CocoEvaluator(
                coco_gt=coco_gt,
                output_dir=os.path.join(self.output_dir, f"dataset_{i}")
            )

            preds = all_preds[idx:idx + num_samples]
            metas = all_metas[idx:idx + num_samples]

            stats = evaluator.evaluate(preds, metas, epoch=epoch)
            results[f"dataset_{i}"] = stats

            idx += num_samples

        return results

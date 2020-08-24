from mmcv.utils import build_from_cfg

from mmdet.datasets.builder import DATASETS, _concat_dataset
from mmdet.datasets.dataset_wrappers import ClassBalancedDataset, ConcatDataset, RepeatDataset

from .datasets.source_balanced_dataset import SourceBalancedDataset


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(build_dataset(cfg["dataset"], default_args), cfg["times"])
    elif cfg["type"] == "ClassBalancedDataset":
        dataset = ClassBalancedDataset(build_dataset(cfg["dataset"], default_args), cfg["oversample_thr"])
    elif cfg["type"] == "SourceBalancedDataset":
        dataset = SourceBalancedDataset(build_dataset(cfg["dataset"], default_args), cfg["oversample_thr"])
    elif isinstance(cfg.get("ann_file"), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset

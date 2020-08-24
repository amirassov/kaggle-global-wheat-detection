import argparse
from collections import OrderedDict
from typing import Dict, List

import torch

from mmdet.datasets.coco import CocoDataset

BACKGROUND_INDEX = 0
NUM_COCO_CLASSES = 80


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", default="/dumps/DetectoRS_X101-ed983634_v2.pth")
    parser.add_argument("--output_path", default="/dumps/DetectoRS_X101-ed983634_v3.pth")
    parser.add_argument("--classes", default=["broccoli"], nargs="+")
    return parser.parse_args()


def change_weights(name: str, w: torch.Tensor, class_indices: List[int]):
    old_shape = w.shape
    if name.startswith("roi_head.bbox_head"):
        if "fc_cls" in name:
            w = w[[BACKGROUND_INDEX] + class_indices]
        elif "fc_reg" in name and w.shape[0] == 4 * NUM_COCO_CLASSES:
            w = w[sum([[i, i + 1, i + 2, i + 3] for i in class_indices], [])]
        print(f"{name}: {old_shape} -> {w.shape}")
    return w


def main():
    args = parse_args()
    class_indices = [CocoDataset.CLASSES.index(cls) for cls in args.classes]
    print(class_indices)
    weights: Dict[str, OrderedDict] = torch.load(args.weights_path, map_location="cpu")
    for name, w in weights["state_dict"].items():
        weights["state_dict"][name] = change_weights(name, w, class_indices)
    torch.save(weights, args.output_path)


if __name__ == "__main__":
    main()

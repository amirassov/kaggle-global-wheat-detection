import argparse
import os.path as osp
from glob import glob

import cv2
import pandas as pd
from tqdm import tqdm

from gwd.converters import kaggle2coco


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-pattern", default="/data/SPIKE_images/*jpg")
    parser.add_argument("--annotation-root", default="/data/SPIKE_annotations")
    parser.add_argument("--kaggle_output_path", default="/data/spike.csv")
    parser.add_argument("--coco_output_path", default="/data/coco_spike.json")
    return parser.parse_args()


def main():
    args = parse_args()
    img_paths = glob(args.image_pattern)
    annotations = []
    for img_path in tqdm(img_paths):
        ann_path = osp.join(args.annotation_root, (osp.basename(img_path.replace("jpg", "bboxes.tsv"))))
        ann = pd.read_csv(ann_path, sep="\t", names=["x_min", "y_min", "x_max", "y_max"])
        h, w = cv2.imread(img_path).shape[:2]
        ann[["x_min", "x_max"]] = ann[["x_min", "x_max"]].clip(0, w)
        ann[["y_min", "y_max"]] = ann[["y_min", "y_max"]].clip(0, h)
        ann["height"] = h
        ann["width"] = w
        ann["bbox_width"] = ann["x_max"] - ann["x_min"]
        ann["bbox_height"] = ann["y_max"] - ann["y_min"]
        ann = ann[(ann["bbox_width"] > 0) & (ann["bbox_height"] > 0)].copy()
        ann["bbox"] = ann[["x_min", "y_min", "bbox_width", "bbox_height"]].values.tolist()
        ann["image_id"] = osp.basename(img_path).split(".")[0]
        annotations.append(ann)
    annotations = pd.concat(annotations)
    annotations["source"] = "spike"
    print(annotations.head())
    annotations[["image_id", "source", "width", "height", "bbox"]].to_csv(args.kaggle_output_path, index=False)
    kaggle2coco.main(args.kaggle_output_path, args.coco_output_path)


if __name__ == "__main__":
    main()

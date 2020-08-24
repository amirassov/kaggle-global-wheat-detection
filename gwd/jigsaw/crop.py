import argparse
import os
import os.path as osp
from functools import partial
from multiprocessing import Pool

import cv2
import pandas as pd
from tqdm import tqdm

from gwd.converters import images2coco

CROP_SIZE = 1024
OFFSET_SIZE = 512


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-path", default="/data/folds_v2/0/mosaic_train.csv")
    parser.add_argument("--img-root", default="/data/mosaics")
    parser.add_argument("--output-root", default="/data/crops_fold0")
    parser.add_argument("--output-path", default="/data/coco_crops_fold0.json")
    return parser.parse_args()


def crop_and_save(img_path, output_root):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    if h <= 1024 and w <= 1024:
        return

    for i in range(0, h // OFFSET_SIZE - 1):
        for j in range(0, w // OFFSET_SIZE - 1):
            if i % 2 or j % 2:
                crop = img[i * OFFSET_SIZE : i * OFFSET_SIZE + CROP_SIZE, j * OFFSET_SIZE : j * OFFSET_SIZE + CROP_SIZE]
                img_name = osp.basename(img_path)
                crop_path = osp.join(output_root, f"{i}_{j}_{img_name}")
                cv2.imwrite(crop_path, crop)


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    annotations = pd.read_csv(args.annotation_path)
    annotations["img_path"] = annotations["image_id"].apply(lambda x: f"{args.img_root}/{x}.jpg")
    img_paths = annotations["img_path"].drop_duplicates().tolist()
    with Pool(32) as p:
        list(
            tqdm(iterable=p.imap(partial(crop_and_save, output_root=args.output_root), img_paths), total=len(img_paths))
        )
    images2coco.main(img_pattern=osp.join(args.output_root, "*.jpg"), output_path=args.output_path)


if __name__ == "__main__":
    main()

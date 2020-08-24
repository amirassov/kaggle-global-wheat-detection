import argparse
import os
import os.path as osp

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from gwd.eda.visualization import draw_bounding_boxes_on_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_path", default="/data/pseudo_universe_submission.csv")
    parser.add_argument("--img_root", default="/data/test")
    parser.add_argument("--output_root", default="/data/eda")
    return parser.parse_args()


def convert_bboxes(bboxes):
    bboxes = np.concatenate([bboxes[:, 1:], bboxes[:, :1]], axis=1)
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    submission = pd.read_csv(args.submission_path)
    for _, row in tqdm(submission.iterrows(), total=len(submission)):
        image = cv2.imread(osp.join(args.img_root, f"{row.image_id}.jpg"))
        bboxes = convert_bboxes(np.array(list(map(float, row.PredictionString.split()))).reshape(-1, 5))
        draw_bounding_boxes_on_image(image, bboxes, use_normalized_coordinates=False, thickness=5)
        cv2.imwrite(osp.join(args.output_root, f"{row.image_id}.jpg"), image)


if __name__ == "__main__":
    main()

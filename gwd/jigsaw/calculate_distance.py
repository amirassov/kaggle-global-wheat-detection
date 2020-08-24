"""
Based on https://github.com/lRomul/argus-tgs-salt/blob/master/mosaic/create_mosaic.py
"""
import argparse
import os
import os.path as osp
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-path", default="/data/train.csv")
    parser.add_argument("--img-root", default="/data/train")
    parser.add_argument("--output-root", default="/data/distances")
    return parser.parse_args()


def get_descriptor(image_path):
    img = cv2.imread(image_path).astype(float)
    return [(img[:, 0], img[:, 1]), (img[:, -2], img[:, -1]), (img[0, :], img[1, :]), (img[-2, :], img[-1, :])]


def get_descriptors(image_paths):
    with Pool(32) as p:
        descriptors = list(
            tqdm(iterable=p.imap(get_descriptor, image_paths), total=len(image_paths), desc="Get descriptors...")
        )
    return descriptors


def calc_pair_metric(args):
    i, desc_i, j, desc_j = args
    if i != j:
        left_metric = calc_metric(desc_i[0], desc_j[1])
        top_metric = calc_metric(desc_i[2], desc_j[3])
    else:
        left_metric = top_metric = 1e6
    return left_metric, top_metric


def norm(a):
    return np.mean(a ** 2)


def calc_metric(d1, d2):
    """
       d2    |  d1
    [-2  -1] | [0 1]
    """
    return (norm((d1[1] + d2[1] - 2 * d1[0]) / 2) + norm((d2[0] + d1[0] - 2 * d2[1]) / 2)) / 2


def get_metrics(descriptors):
    n_samples = len(descriptors)
    left_matrix = np.zeros((n_samples, n_samples))
    top_matrix = np.zeros((n_samples, n_samples))
    for i, desc_i in tqdm(enumerate(descriptors), total=n_samples, desc="Get metrics..."):
        with Pool(32) as p:
            metrics = list((p.imap(calc_pair_metric, [(i, desc_i, j, desc_j) for j, desc_j in enumerate(descriptors)])))
        for j, metric in enumerate(metrics):
            left_matrix[i, j], top_matrix[i, j] = metric
    return left_matrix, top_matrix


def get_distances(image_paths):
    descriptors = get_descriptors(image_paths)
    left_matrix, top_matrix = get_metrics(descriptors)
    return left_matrix, top_matrix


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    annotations = pd.read_csv(args.annotation_path).drop_duplicates("image_id")
    for source, source_annotations in annotations.groupby("source"):
        print(f"Source: {source}")
        image_paths = source_annotations["image_id"].apply(lambda x: osp.join(args.img_root, f"{x}.jpg")).tolist()
        left_matrix, top_matrix = get_distances(image_paths)
        np.savez(
            osp.join(args.output_root, f"{source}.npz"),
            left_matrix=left_matrix,
            top_matrix=top_matrix,
            paths=image_paths,
        )


if __name__ == "__main__":
    main()

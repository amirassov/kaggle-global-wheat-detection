import argparse
import ast
import os
import os.path as osp

import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

SOURCES = ["ethz_1", "arvalis_1", "arvalis_3", "usask_1", "rres_1", "inrae_1", "arvalis_2"]
VAL_SOURCES = ["usask_1", "ethz_1"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mosaic_path", default="/data/train_mosaic.csv")
    parser.add_argument("--annotation_path", default="/data/train.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--output_root", default="/data/folds_v2")
    return parser.parse_args()


def save_split(annotations, val_ids, output_root, prefix):
    os.makedirs(output_root, exist_ok=True)
    train = annotations[~annotations["image_id"].isin(val_ids)]
    val = annotations[annotations["image_id"].isin(val_ids)]
    print(f"{prefix} train length: {len(set(train['image_id']))}")
    print(f"{prefix} val length: {len(set(val['image_id']))}\n")
    train.to_csv(osp.join(output_root, f"{prefix}_train.csv"), index=False)
    val.to_csv(osp.join(output_root, f"{prefix}_val.csv"), index=False)


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    tile_annotations = pd.read_csv(args.annotation_path)
    mosaic_annotations = pd.read_csv(args.mosaic_path, converters={"bbox": ast.literal_eval})
    mosaic_annotations["num_of_bboxes"] = mosaic_annotations["image_id"].map(
        mosaic_annotations["image_id"].value_counts()
    )
    mosaic_annotations["median_area"] = mosaic_annotations["bbox"].apply(lambda x: np.sqrt(x[-1] * x[-2]))
    mosaic_annotations["source_index"] = mosaic_annotations["source"].apply(lambda x: SOURCES.index(x))
    images = (
        mosaic_annotations[["image_id", "source_index", "median_area", "num_of_bboxes", "source"]]
        .copy()
        .drop_duplicates("image_id")
    )
    images = images[~images["source"].isin(VAL_SOURCES)]
    splitter = MultilabelStratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=3)
    for i, (train_index, test_index) in enumerate(
        splitter.split(images, images[["source_index", "median_area", "num_of_bboxes"]])
    ):
        mosaic_val_ids = images.iloc[test_index, images.columns.get_loc("image_id")]
        tile_val_ids = sum([x.split("_") for x in mosaic_val_ids], [])

        fold_root = osp.join(args.output_root, str(i))

        save_split(mosaic_annotations, mosaic_val_ids, fold_root, prefix="mosaic")
        save_split(tile_annotations, tile_val_ids, fold_root, prefix="tile")


if __name__ == "__main__":
    main()

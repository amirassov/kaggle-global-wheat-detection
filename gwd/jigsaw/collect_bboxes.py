import argparse
import ast
import os.path as osp

import mmcv
import pandas as pd
from tqdm import tqdm

IMG_SIZE = 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mosaics-path", default="/data/mosaics.json")
    parser.add_argument("--annotation-path", default="/data/train.csv")
    parser.add_argument("--output-path", default="/data/train_mosaic.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = pd.read_csv(args.annotation_path, converters={"bbox": ast.literal_eval})
    annotations["tile_id"] = 0
    mosaics = mmcv.load(args.mosaics_path)
    for mosaic in tqdm(mosaics):
        mosaic_image_id = "_".join([osp.basename(tile["path"]).split(".")[0] for tile in mosaic])
        for i, tile in enumerate(mosaic):
            image_id = osp.basename(tile["path"]).split(".")[0]
            mask = annotations["image_id"] == image_id
            annotations.loc[mask, "bbox"] = annotations.loc[mask, "bbox"].apply(
                lambda bbox: [bbox[0] + IMG_SIZE * tile["i"], bbox[1] + IMG_SIZE * tile["j"], bbox[2], bbox[3]]
            )
            annotations.loc[mask, "tile_id"] = i
            annotations.loc[mask, "image_id"] = mosaic_image_id
        mosaic_width = (max([x["i"] for x in mosaic]) + 1) * IMG_SIZE
        mosaic_height = (max([x["j"] for x in mosaic]) + 1) * IMG_SIZE
        annotations.loc[annotations["image_id"] == mosaic_image_id, "width"] = mosaic_width
        annotations.loc[annotations["image_id"] == mosaic_image_id, "height"] = mosaic_height
    print(annotations.head())
    annotations.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()

import argparse
import os
import os.path as osp
from shutil import copyfile

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", default="/data/coco_train.json")
    parser.add_argument("--image_root", default="/data/train")
    parser.add_argument("--output_root", default="/data/sources")
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = mmcv.load(args.annotation_path)
    for sample in tqdm(annotations["images"]):
        source_root = osp.join(args.output_root, sample["source"])
        os.makedirs(source_root, exist_ok=True)
        copyfile(osp.join(args.image_root, sample["file_name"]), osp.join(source_root, sample["file_name"]))


if __name__ == "__main__":
    main()

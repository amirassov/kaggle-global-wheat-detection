import argparse
import os.path as osp
from glob import glob
from multiprocessing import Pool

import cv2
import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_pattern", default="/data/test/*jpg")
    parser.add_argument("--output_path", default="/data/coco_test.json")
    parser.add_argument("--n_jobs", type=int, default=16)
    return parser.parse_args()


def convert(id_path):
    img_id, img_path = id_path
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    return {"id": img_id, "height": h, "width": w, "file_name": osp.basename(img_path)}


def main(img_pattern, output_path, n_jobs=16):
    img_paths = glob(img_pattern)
    with Pool(n_jobs) as p:
        coco_images = list(
            tqdm(
                iterable=p.imap_unordered(convert, enumerate(img_paths)), total=len(img_paths), desc="Images to COCO..."
            )
        )

    mmcv.dump(
        {
            "annotations": [],
            "images": coco_images,
            "categories": [{"supercategory": "wheat", "name": "wheat", "id": 1}],
        },
        output_path,
    )


if __name__ == "__main__":
    main(**vars(parse_args()))

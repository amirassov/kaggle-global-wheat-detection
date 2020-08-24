import argparse
import os
import os.path as osp

import mmcv
from pycocotools.coco import COCO
from tqdm import tqdm

SCORE_THRESHOLD = 0.8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", default="/data/test")
    parser.add_argument("--annotation_path", default="/data/rfp_r50_ga_mstrain_pseudo_stage1_test_predictions.json")
    parser.add_argument("--output_root", default="/data/test_wheat_crops")
    parser.add_argument("--from_predict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    dataset = COCO(args.annotation_path)
    for img_id, img_info in tqdm(dataset.imgs.items()):
        filename = img_info["file_name"]
        img = mmcv.imread(osp.join(args.img_root, filename))
        ann_ids = dataset.getAnnIds(imgIds=[img_id])
        ann_info = dataset.loadAnns(ann_ids)
        for i, ann in enumerate(ann_info):
            x1, y1, w, h = map(int, ann["bbox"])
            category_id = ann["category_id"]
            crop = img[y1 : y1 + h, x1 : x1 + w]
            output_path = osp.join(args.output_root, f"{x1}_{y1}_{w}_{h}_{filename}")
            if w > 6 and h > 6:
                if args.from_predict:
                    if category_id == 2 and ann["score"] > SCORE_THRESHOLD:  # prediction
                        mmcv.imwrite(crop, output_path)
                elif category_id == 1 and img_info["source"] in ["arvalis_1", "arvalis_3", "rres_1", "inrae_1"]:
                    mmcv.imwrite(crop, output_path)


if __name__ == "__main__":
    main()

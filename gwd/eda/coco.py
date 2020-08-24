import argparse
import os
import os.path as osp
import tempfile

import cv2
import mmcv
from pycocotools.coco import COCO
from tqdm import tqdm

from gwd.eda.visualization import draw_bounding_boxes_on_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-prefix", default="/data/train")
    parser.add_argument("--ann-file", default="/data/rfp_r50_ga_mstrain_pseudo_stage1_epoch_10_train_predictions.json")
    parser.add_argument("--prediction_path")
    parser.add_argument("--output_root", type=str, default="/data/eda")
    return parser.parse_args()


def prepare_predictions(prediction_path, annotation_path, output_path):
    predictions = mmcv.load(prediction_path)
    annotations = mmcv.load(annotation_path)
    for i, prediction in tqdm(enumerate(predictions), total=len(predictions)):
        prediction["segmentation"] = ""
        x1, y1, w, h = prediction["bbox"]
        prediction["area"] = w * h
        prediction["id"] = i
        prediction["iscrowd"] = 0
    annotations["annotations"] = predictions
    mmcv.dump(annotations, output_path)


def prepare_bboxes(ann_info):
    bboxes = []
    labels = []
    display_str_list = []
    for i, ann in enumerate(ann_info):
        x1, y1, w, h = ann["bbox"]
        if ann["area"] <= 0 or w < 1 or h < 1:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        bboxes.append(bbox)
        labels.append(ann["category_id"])
        if "score" in ann:
            display_str_list.append(f'{ann["score"]:.2f}')
        else:
            display_str_list.append("")
    return bboxes, labels, display_str_list


def main():
    args = parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    if args.prediction_path is not None:
        with tempfile.TemporaryDirectory() as root:
            tmp_ann_file = osp.join(root, "ann.json")
            prepare_predictions(args.prediction_path, args.ann_file, tmp_ann_file)
            dataset = COCO(tmp_ann_file)
    else:
        dataset = COCO(args.ann_file)

    for img_id, img_info in tqdm(dataset.imgs.items()):
        img = cv2.imread(osp.join(args.img_prefix, img_info["file_name"]))[..., ::-1]
        ann_ids = dataset.getAnnIds(imgIds=[img_id])
        ann_info = dataset.loadAnns(ann_ids)
        bboxes, labels, display_str_list = prepare_bboxes(ann_info)
        label2colors = {
            -1: {"bbox": (255, 0, 0)},
            1: {"bbox": (0, 128, 255)},
            2: {"bbox": (128, 0, 0), "text": (255, 255, 255)},
        }
        draw_bounding_boxes_on_image(
            img,
            bboxes,
            labels=labels,
            label2colors=label2colors,
            display_str_list=display_str_list,
            use_normalized_coordinates=False,
            thickness=4,
            fontsize=15,
        )
        filename = f"{img_info['file_name']}"
        if "ap" in img_info:
            filename = f"{img_info['ap']:.2f}_{filename}"
        cv2.imwrite(osp.join(args.output_root, filename), img[..., ::-1])


if __name__ == "__main__":
    main()

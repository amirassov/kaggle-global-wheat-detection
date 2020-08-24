import argparse
import ast
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import mmcv
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", default="/data/folds_v2/0/clean_tile_train.csv")
    parser.add_argument("--output_path", default="/data/folds_v2/0/coco_clean_tile_train.json")
    return parser.parse_args()


def group2coco(image_name: str, group: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    coco_annotations = []
    for _, row in group.iterrows():
        x_min, y_min, width, height = row["bbox"]
        category_id = 1
        is_ignore = row.get("ignore", False)
        coco_annotations.append(
            {
                "segmentation": "",
                "area": float(width * height),
                "category_id": category_id,
                "bbox": [float(x_min), float(y_min), float(width), float(height)],
                "iscrowd": int(is_ignore),
            }
        )

    return (
        {
            "width": int(group["width"].iloc[0]),
            "height": int(group["height"].iloc[0]),
            "file_name": image_name,
            "source": group["source"].iloc[0],
        },
        coco_annotations,
    )


def main(annotation_path, output_path, exclude_sources=None):
    annotations = pd.read_csv(annotation_path, converters={"bbox": ast.literal_eval})
    if exclude_sources is not None:
        annotations = annotations[~annotations["source"].isin(exclude_sources)]
    annotations["image_name"] = annotations["image_id"].apply(lambda x: f"{x}.jpg")

    coco_annotations = []
    coco_images = []
    image_groups = annotations.groupby("image_id")

    for i, (image_name, group) in tqdm(enumerate(annotations.groupby("image_name")), total=len(image_groups)):
        image_info, image_annotations = group2coco(image_name, group)

        image_info["id"] = i
        for ann in image_annotations:
            ann["image_id"] = i
        coco_images.append(deepcopy(image_info))
        coco_annotations.extend(deepcopy(image_annotations))

    for i, ann in enumerate(coco_annotations):
        ann["id"] = i

    print(f"Length of images: {len(coco_images)}")
    print(f"Length of annotations: {len(coco_annotations)}")
    print(f"Length set image id: {len(set([x['id'] for x in coco_images]))}")
    print(f"Max image id: {max([x['id'] for x in coco_images])}")
    print(coco_images[0])
    print(coco_annotations[0])

    mmcv.dump(
        {
            "annotations": coco_annotations,
            "images": coco_images,
            "categories": [{"supercategory": "wheat", "name": "wheat", "id": 1}],
        },
        output_path,
    )


if __name__ == "__main__":
    main(**vars(parse_args()))

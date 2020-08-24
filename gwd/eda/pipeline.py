import argparse
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config
from tqdm import tqdm

from gwd.datasets.wheat_detection import WheatDataset
from gwd.eda.kmeans import kmeans
from gwd.eda.visualization import draw_bounding_boxes_on_image
from mmdet.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/rfp_spike/rfp_r50_ga_mstrain_spike_stage1.py")
    parser.add_argument("--output_root", default="/data/eda")
    parser.add_argument("--fold", default=0)
    parser.add_argument("--skip_type", nargs="+", default=("DefaultFormatBundle", "Normalize", "Collect"))
    return parser.parse_args()


def retrieve_data_cfg(config_path, fold, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    if "ann_file" not in cfg.data.train:
        train_data_cfg = train_data_cfg.dataset
    if isinstance(cfg.data.train.ann_file, list):
        cfg.data.train.ann_file = [x.format(fold=fold) for x in cfg.data.train.ann_file]
    elif isinstance(cfg.data.train.ann_file, str):
        cfg.data.train.ann_file = cfg.data.train.ann_file.format(fold=fold)
    train_data_cfg["pipeline"] = [x for x in train_data_cfg.pipeline if x["type"] not in skip_type]
    return cfg


def plot_statistics(widths, heights, output_root):
    statistics = {
        "width": {"array": widths / 1024, "axis": (0, 0.5)},
        "height": {"array": heights / 1024, "axis": (0, 0.5)},
        "area": {"array": widths * heights / 1024 / 1024, "axis": (0, 0.1)},
        "ratio": {"array": widths / heights, "axis": (0, 4)},
    }
    axes = {}
    fig, ((axes["width"], axes["height"]), (axes["area"], axes["ratio"])) = plt.subplots(2, 2, figsize=(12, 12))
    for name, data in statistics.items():
        axes[name].hist(data["array"], 250, density=True)
        axes[name].set_title(f"{name}")
        axes[name].grid(True)
        axes[name].set_xlim(data["axis"])
    fig.savefig(osp.join(output_root, "statistics.png"))


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    cfg = retrieve_data_cfg(args.config_path, args.fold, args.skip_type)

    dataset: WheatDataset = build_dataset(cfg.data.train)
    from IPython import embed

    embed()
    heights = []
    widths = []
    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        image_id = osp.basename(dataset.data_infos[i]["file_name"])
        image = data["img"]
        bboxes = data["gt_bboxes"]
        ignore_bboxes = data["gt_bboxes_ignore"]
        if len(bboxes) == 0:
            print(image.shape)
        widths.append(bboxes[:, 2] - bboxes[:, 0])
        heights.append(bboxes[:, 3] - bboxes[:, 1])
        draw_bounding_boxes_on_image(image, bboxes, use_normalized_coordinates=False, thickness=5)
        if len(ignore_bboxes):
            draw_bounding_boxes_on_image(
                image,
                ignore_bboxes,
                label2colors={None: {"bbox": (0, 255, 0)}},
                use_normalized_coordinates=False,
                thickness=5,
            )
        cv2.imwrite(osp.join(args.output_root, f"{i}_{image_id}"), image)
    widths = np.concatenate(widths)
    heights = np.concatenate(heights)
    clusters = kmeans(np.stack([heights, widths], axis=1), k=10)
    print(f"aspect rations: {clusters[:, 0] / clusters[:, 1]}")
    print(f"sizes: {np.sqrt(clusters[:, 0] * clusters[:, 1])}")
    plot_statistics(widths, heights, args.output_root)


if __name__ == "__main__":
    main()

import argparse
import os
from typing import Tuple

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import print_log

from gwd.datasets.wheat_detection import WheatDataset
from gwd.misc.logging import get_logger
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("--config", help="test config file path")
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument("--img-prefix")
    parser.add_argument("--ann-file")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where painted images will be saved")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--show-score-thr", type=float, default=0.3, help="score threshold (default: 0.3)")
    parser.add_argument("--gpu-collect", action="store_true", help="whether to use gpu to collect results.")
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--options", nargs="+", action=DictAction, help="arguments in dict")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="pytorch", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--score_thr", type=float)
    parser.add_argument("--iou_thr", type=float)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main(
    config=None,
    checkpoint=None,
    img_prefix=None,
    ann_file=None,
    flip=False,
    out=None,
    format_only=False,
    eval=None,
    show=False,
    show_dir=None,
    log_file=None,
    fold=None,
    show_score_thr=0.3,
    gpu_collect=False,
    tmpdir=None,
    options=None,
    launcher="none",
    local_rank=0,
    score_thr=None,
    iou_thr=None,
) -> Tuple[list, WheatDataset]:
    logger = get_logger("inference", log_file=log_file, log_mode="a")
    assert out or eval or format_only or show or show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if eval and format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if out is not None and not out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(config)
    if fold is not None:
        cfg.data.test.ann_file = cfg.data.test.ann_file.format(fold=fold)
    if img_prefix is not None:
        cfg.data.test.img_prefix = img_prefix
    if ann_file is not None:
        cfg.data.test.ann_file = ann_file
    if flip:
        print(cfg.data.test.pipeline[-1])
        print(f"flip: {cfg.data.test.pipeline[-1].flip}")
        cfg.data.test.pipeline[-1].flip_direction = ["horizontal", "vertical"]
        cfg.data.test.pipeline[-1].flip = flip
    if score_thr is not None:
        if "rcnn" in cfg.test_cfg:
            cfg.test_cfg.rcnn.score_thr = score_thr
        else:
            cfg.test_cfg.score_thr = score_thr
    if iou_thr is not None:
        if "rcnn" in cfg.test_cfg:
            cfg.test_cfg.rcnn.nms.iou_thr = iou_thr
        else:
            cfg.test_cfg.nms.iou_thr = iou_thr
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=distributed, shuffle=False
    )

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint["meta"]:
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, show, show_dir, show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
        outputs = multi_gpu_test(model, data_loader, tmpdir, gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if out:
            print(f"\nwriting results to {out}")
            mmcv.dump(outputs, out)
        kwargs = {} if options is None else options
        if format_only:
            dataset.format_results(outputs, **kwargs)
        if eval:
            print_log(f"config: {config}", logger)
            print_log(f"score_thr: {score_thr}", logger)
            print_log(f"iou_thr: {iou_thr}", logger)
            dataset.evaluate(outputs, logger=logger, **kwargs)
    return outputs, dataset


if __name__ == "__main__":
    main(**vars(parse_args()))

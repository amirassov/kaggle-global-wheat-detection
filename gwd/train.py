import argparse
import copy
import os
import os.path as osp
import time
import warnings  # noqa E402

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist

import gwd  # noqa F401
from gwd.patches import build_dataset
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

warnings.simplefilter("ignore", UserWarning)  # noqa E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument("--load-from", help="the checkpoint file to load from")
    parser.add_argument(
        "--no-validate", action="store_true", help="whether not to evaluate the checkpoint during training"
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus", type=int, help="number of gpus to use " "(only applicable to non-distributed training)"
    )
    group_gpus.add_argument(
        "--gpu-ids", type=int, nargs="+", help="ids of gpus to use " "(only applicable to non-distributed training)"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic", action="store_true", help="whether to set deterministic options for CUDNN backend."
    )
    parser.add_argument("--options", nargs="+", action=DictAction, help="arguments in dict")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main(
    config=None,
    fold=None,
    work_dir=None,
    resume_from=None,
    load_from=None,
    no_validate=False,
    gpus=None,
    gpu_ids=None,
    seed=None,
    deterministic=False,
    options=None,
    launcher="none",
    local_rank=0,
):
    cfg = Config.fromfile(config)

    if fold is not None:
        if "ann_file" in cfg.data.train:
            if isinstance(cfg.data.train.ann_file, list):
                cfg.data.train.ann_file = [x.format(fold=fold) for x in cfg.data.train.ann_file]
            elif isinstance(cfg.data.train.ann_file, str):
                cfg.data.train.ann_file = cfg.data.train.ann_file.format(fold=fold)
        else:
            cfg.data.train.dataset.ann_file = cfg.data.train.dataset.ann_file.format(fold=fold)
        cfg.data.val.ann_file = cfg.data.val.ann_file.format(fold=fold)
        cfg.data.test.ann_file = cfg.data.test.ann_file.format(fold=fold)
    if options is not None:
        cfg.merge_from_dict(options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join("/dumps/work_dirs", osp.splitext(osp.basename(config))[0], str(fold))
    if resume_from is not None:
        cfg.resume_from = resume_from
    if load_from is not None:
        cfg.load_from = load_from
    if gpu_ids is not None:
        cfg.gpu_ids = gpu_ids
    else:
        cfg.gpu_ids = range(1) if gpus is None else range(gpus)

    # init distributed env first, since logger depends on the dist info.
    if launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if seed is not None:
        logger.info(f"Set random seed to {seed}, " f"deterministic: {deterministic}")
        set_random_seed(seed, deterministic=deterministic)
    cfg.seed = seed
    meta["seed"] = seed

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.pretty_text, CLASSES=datasets[0].CLASSES
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model, datasets, cfg, distributed=distributed, validate=(not no_validate), timestamp=timestamp, meta=meta
    )


if __name__ == "__main__":
    main(**vars(parse_args()))

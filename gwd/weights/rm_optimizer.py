import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="/dumps/work_dirs/universe_r101_gfl_mstrain_stage1/0/epoch_11.pth")
    parser.add_argument(
        "--output",
        default="/dumps/work_dirs/universe_r101_gfl_mstrain_stage1/0/universe_r101_gfl_mstrain_stage1_epoch_11.pth",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights = torch.load(args.weights, map_location="cpu")
    del weights["optimizer"]
    torch.save(weights, args.output)


if __name__ == "__main__":
    main()

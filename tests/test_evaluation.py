import numpy as np
import pytest
from mmcv import Config

from gwd.datasets.evaluation import calc_tpfpfn, kaggle_map
from mmdet.datasets import build_dataset


@pytest.fixture(scope="session")
def dataset():
    cfg = Config.fromfile("configs/_base_/datasets/wheat_detection.py")
    return build_dataset(cfg)


@pytest.fixture(scope="session")
def gt_bboxes():
    return np.array([[954, 391, 1024, 481], [660, 220, 755, 322]])


@pytest.fixture(scope="session")
def one_bboxes():
    return np.array([[954, 391, 1024, 481, 0]])


@pytest.fixture(scope="session")
def annotations(gt_bboxes):
    return [{"bboxes": gt_bboxes, "labels": np.zeros(len(gt_bboxes))}]


def test_non_overlapping(gt_bboxes):
    det_bboxes = np.array([[0, 0, 10, 10, 0]])
    tp, fp, fn = calc_tpfpfn(det_bboxes, gt_bboxes)
    assert tp == 0
    assert fp == 1
    assert fn == len(gt_bboxes)


def test_one(gt_bboxes, one_bboxes):
    tp, fp, fn = calc_tpfpfn(one_bboxes, gt_bboxes)
    assert tp == 1
    assert fp == 0
    assert fn == len(gt_bboxes) - 1


def test_empty_map(annotations):
    mean_ap, _ = kaggle_map([[np.array([]).reshape(-1, 5)]], annotations, iou_thrs=[0.5])
    assert mean_ap == 0


def test_one_map(annotations, one_bboxes):
    mean_ap, _ = kaggle_map([[one_bboxes]], annotations, iou_thrs=[0.5])
    assert mean_ap == 1 / 2


def test_map():
    gt_bboxes = np.array(
        [
            [954, 391, 1024, 481],
            [660, 220, 755, 322],
            [64, 209, 140, 266],
            [896, 99, 998, 168],
            [747, 460, 819, 537],
            [885, 163, 988, 232],
            [514, 399, 604, 496],
            [702, 794, 799, 893],
            [721, 624, 819, 732],
            [826, 512, 908, 606],
            [883, 944, 962, 1018],
            [247, 594, 370, 686],
            [673, 514, 768, 627],
            [829, 847, 931, 957],
            [94, 737, 186, 844],
            [588, 568, 663, 675],
            [158, 890, 261, 954],
            [744, 906, 819, 985],
            [826, 33, 898, 107],
            [601, 69, 668, 156],
        ]
    )
    annotations = [{"bboxes": gt_bboxes, "labels": np.zeros(len(gt_bboxes))}]
    det_bboxes = np.array(
        [
            [956.0, 409.0, 1024.0, 494.0, 0.997],
            [883.0, 945.0, 968.0, 1022.0, 0.996],
            [745.0, 468.0, 826.0, 555.0, 0.995],
            [658.0, 239.0, 761.0, 344.0, 0.994],
            [518.0, 419.0, 609.0, 519.0, 0.993],
            [711.0, 805.0, 803.0, 911.0, 0.992],
            [62.0, 213.0, 134.0, 277.0, 0.991],
            [884.0, 175.0, 993.0, 243.0, 0.99],
            [721.0, 626.0, 817.0, 730.0, 0.98],
            [878.0, 619.0, 999.0, 700.0, 0.97],
            [887.0, 107.0, 998.0, 178.0, 0.95],
            [827.0, 525.0, 915.0, 608.0, 0.94],
            [816.0, 868.0, 918.0, 954.0, 0.93],
            [166.0, 882.0, 244.0, 957.0, 0.92],
            [603.0, 563.0, 681.0, 660.0, 0.91],
            [744.0, 916.0, 812.0, 968.0, 0.89],
            [582.0, 86.0, 668.0, 158.0, 0.88],
            [79.0, 715.0, 170.0, 816.0, 0.86],
            [246.0, 586.0, 341.0, 666.0, 0.85],
            [181.0, 512.0, 274.0, 601.0, 0.84],
            [655.0, 527.0, 754.0, 617.0, 0.80],
            [568.0, 363.0, 629.0, 439.0, 0.79],
            [9.0, 717.0, 161.0, 827.0, 0.74],
            [576.0, 698.0, 651.0, 776.0, 0.60],
            [805.0, 974.0, 880.0, 1024.0, 0.59],
            [10.0, 15.0, 88.0, 79.0, 0.55],
            [826.0, 40.0, 895.0, 114.0, 0.53],
            [32.0, 983.0, 138.0, 1023.0, 0.50],
        ]
    )
    assert abs(kaggle_map([[det_bboxes]], annotations, iou_thrs=[0.5])[0] - 0.6552) < 1e-3
    assert abs(kaggle_map([[det_bboxes]], annotations, iou_thrs=[0.75])[0] - 0.0909) < 1e-3
    assert abs(kaggle_map([[det_bboxes]], annotations, iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75])[0] - 0.3663) < 1e-3

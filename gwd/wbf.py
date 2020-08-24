import argparse
from functools import partial
from multiprocessing import Pool

import mmcv
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from scipy.stats import rankdata
from tqdm import tqdm

import gwd  # noqa F401
from mmdet.datasets import build_dataset

WHEAT_CLASS_ID = 0
IMAGE_SIZE = 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_paths",
        default=[
            "/data/rfp_r50_ga_mstrain_stage2_fold0_predictions.pkl",
            "/data/universe_r50_mstrain_spike_stage1_epoch_12_predictions.pkl",
        ],
        nargs="+",
    )
    parser.add_argument("--iou_thr", default=0.55, type=float)
    parser.add_argument("--score_thr", default=0.45, type=float)
    parser.add_argument("--weights", default=[1.0, 1.0], type=float, nargs="+")
    return parser.parse_args()


def mmdet2wbf(prediction):
    wheat_prediction = prediction[WHEAT_CLASS_ID]
    bboxes = wheat_prediction[:, :4] / IMAGE_SIZE
    scores = np.clip(wheat_prediction[:, 4], 0, 1.0)
    scores = 0.5 * (rankdata(scores) / len(scores)) + 0.5
    labels = np.zeros_like(scores)
    return bboxes, scores, labels


def wbf_per_sample(sample_predictions, weights, iou_thr, score_thr):
    bboxes_list = []
    scores_list = []
    labels_list = []
    for prediction in sample_predictions:
        bboxes, scores, labels = mmdet2wbf(prediction)
        bboxes_list.append(bboxes)
        scores_list.append(scores)
        labels_list.append(labels)
    bboxes, scores, labels = weighted_boxes_fusion(
        boxes_list=bboxes_list,
        scores_list=scores_list,
        labels_list=labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=score_thr,
    )
    return [np.concatenate([bboxes * IMAGE_SIZE, scores.reshape(-1, 1)], axis=1)]


def main(all_predictions, weights, iou_thr, score_thr):
    with Pool(32) as p:
        results = list(
            (
                tqdm(
                    p.imap(
                        partial(wbf_per_sample, weights=weights, iou_thr=iou_thr, score_thr=score_thr),
                        zip(*all_predictions),
                    ),
                    total=len(all_predictions[0]),
                )
            )
        )
    return results


if __name__ == "__main__":
    from mmcv import Config

    cfg = Config.fromfile("configs/_base_/datasets/wheat_detection_mstrain.py")
    cfg.data.test.ann_file = cfg.data.test.ann_file.format(fold=0)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    args = vars(parse_args())
    prediction_paths = args.pop("prediction_paths")
    _all_predictions = [mmcv.load(path) if isinstance(path, str) else path for path in prediction_paths]
    for predictions in _all_predictions:
        dataset.evaluate(results=predictions)

    metrics = {}
    # for _iou_thr in np.arange(0.3, 0.7, 0.1):
    #     for _score_thr in np.arange(0.45, 0.7, 0.1):
    #         for w in np.arange(0.1, 1.0, 0.1):
    for _iou_thr in [0.55]:
        for _score_thr in [0.45]:
            for w in [0.75]:
                _weights = [w, 1 - w]
                print(f"iou_thr: {_iou_thr}, score_thr: {_score_thr}, weights: {_weights}")
                metrics[(_iou_thr, _score_thr, w)] = dataset.evaluate(
                    main(iou_thr=_iou_thr, score_thr=_score_thr, weights=_weights, all_predictions=_all_predictions)
                )
    print(metrics)
    best_parameters = max(metrics, key=metrics.get)
    print(f"best_parameters: {best_parameters}, best_score: {metrics[best_parameters]}")

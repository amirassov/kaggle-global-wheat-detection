from multiprocessing import Pool

import numpy as np
from mmcv.utils import print_log

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.core.evaluation.mean_ap import get_cls_results


def calc_tpfpfn(det_bboxes, gt_bboxes, iou_thr=0.5):
    """Check if detected bboxes are true positive or false positive and if gt bboxes are false negative.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.

    Returns:
        float: (tp, fp, fn).
    """
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    tp = 0
    fp = 0

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        fp = num_dets
        return tp, fp, 0

    ious: np.ndarray = bbox_overlaps(det_bboxes, gt_bboxes)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    gt_covered = np.zeros(num_gts, dtype=bool)
    for i in sort_inds:
        uncovered_ious = ious[i, gt_covered == 0]
        if len(uncovered_ious):
            iou_argmax = uncovered_ious.argmax()
            iou_max = uncovered_ious[iou_argmax]
            if iou_max >= iou_thr:
                gt_covered[[x[iou_argmax] for x in np.where(gt_covered == 0)]] = True
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    fn = (gt_covered == 0).sum()
    return tp, fp, fn


def kaggle_map(
    det_results, annotations, iou_thrs=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75), logger=None, n_jobs=4, by_sample=False
):
    """Evaluate kaggle mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        iou_thrs (list): IoU thresholds to be considered as matched.
            Default: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75).
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        n_jobs (int): Processes used for computing TP, FP and FN.
            Default: 4.
        by_sample (bool): Return AP by sample.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_classes = len(det_results[0])  # positive class num

    pool = Pool(n_jobs)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, _ = get_cls_results(det_results, annotations, i)
        # compute tp and fp for each image with multiple processes
        aps_by_thrs = []
        aps_by_sample = np.zeros(num_imgs)
        for iou_thr in iou_thrs:
            tpfpfn = pool.starmap(calc_tpfpfn, zip(cls_dets, cls_gts, [iou_thr for _ in range(num_imgs)]))
            iou_thr_aps = np.array([tp / (tp + fp + fn) for tp, fp, fn in tpfpfn])
            if by_sample:
                aps_by_sample += iou_thr_aps
            aps_by_thrs.append(np.mean(iou_thr_aps))
        eval_results.append(
            {
                "num_gts": len(cls_gts),
                "num_dets": len(cls_dets),
                "ap": np.mean(aps_by_thrs),
                "ap_by_sample": None if not by_sample else aps_by_sample / len(iou_thrs),
            }
        )
    pool.close()

    aps = []
    for cls_result in eval_results:
        if cls_result["num_gts"] > 0:
            aps.append(cls_result["ap"])
    mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_log(f"\nKaggle mAP: {mean_ap}", logger=logger)
    return mean_ap, eval_results

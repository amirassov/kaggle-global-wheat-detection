import os.path as osp

import mmcv
import numpy as np
import pandas as pd

from gwd.datasets.evaluation import kaggle_map
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


def calc_pseudo_confidence(sample_scores, pseudo_score_threshold):
    if len(sample_scores):
        return np.sum(sample_scores > pseudo_score_threshold) / len(sample_scores)
    else:
        return 0.0


@DATASETS.register_module()
class WheatDataset(CocoDataset):
    CLASSES = ("wheat",)

    def evaluate(self, results, logger=None, iou_thrs=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75), **kwargs):
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        mean_ap, _ = kaggle_map(results, annotations, iou_thrs=iou_thrs, logger=logger)
        return dict(mAP=mean_ap)

    def format_results(self, results, output_path=None, **kwargs):
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )
        prediction_results = []
        for idx in range(len(self)):
            wheat_bboxes = results[idx][0]

            prediction_strs = []
            for bbox in wheat_bboxes:
                x, y, w, h = self.xyxy2xywh(bbox)
                prediction_strs.append(f"{bbox[4]:.4f} {x} {y} {w} {h}")
            filename = self.data_infos[idx]["filename"]
            image_id = osp.splitext(osp.basename(filename))[0]
            prediction_results.append({"image_id": image_id, "PredictionString": " ".join(prediction_strs)})
        predictions = pd.DataFrame(prediction_results)
        if output_path is not None:
            predictions.to_csv(output_path, index=False)
        return predictions

    def evaluate_by_sample(self, results, output_path, logger=None, iou_thrs=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75)):
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        _, eval_results = kaggle_map(results, annotations, iou_thrs=iou_thrs, logger=logger, by_sample=True)
        output_annotations = self.coco.dataset["annotations"]
        output_images = []

        for idx in range(len(self)):
            wheat_bboxes = results[idx][0]
            data_info = self.data_infos[idx]
            data_info["ap"] = eval_results[0]["ap_by_sample"][idx]
            output_images.append(data_info)
            for bbox in wheat_bboxes:
                x, y, w, h = map(float, self.xyxy2xywh(bbox))
                output_annotations.append(
                    {
                        "segmentation": "",
                        "area": w * h,
                        "image_id": data_info["id"],
                        "category_id": 2,
                        "bbox": [x, y, w, h],
                        "iscrowd": 0,
                        "score": float(bbox[-1]),
                    }
                )
        for i, ann in enumerate(output_annotations):
            ann["id"] = i
        outputs = {
            "annotations": output_annotations,
            "images": output_images,
            "categories": [
                {"supercategory": "wheat", "name": "gt", "id": 1},
                {"supercategory": "wheat", "name": "predict", "id": 2},
            ],
        }
        mmcv.dump(outputs, output_path)
        return outputs

    def pseudo_results(self, results, output_path=None, pseudo_score_threshold=0.8, pseudo_confidence_threshold=0.65):
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )
        pseudo_annotations = []
        pseudo_images = []
        for idx in range(len(self)):
            wheat_bboxes = results[idx][0]
            scores = np.array([bbox[-1] for bbox in wheat_bboxes])
            confidence = calc_pseudo_confidence(scores, pseudo_score_threshold=pseudo_score_threshold)
            if confidence >= pseudo_confidence_threshold:
                data_info = self.data_infos[idx]
                data_info["confidence"] = confidence
                pseudo_images.append(data_info)
                for bbox in wheat_bboxes:
                    x, y, w, h = self.xyxy2xywh(bbox)
                    pseudo_annotations.append(
                        {
                            "segmentation": "",
                            "area": w * h,
                            "image_id": data_info["id"],
                            "category_id": 1,
                            "bbox": [x, y, w, h],
                            "iscrowd": 0,
                        }
                    )
        for i, ann in enumerate(pseudo_annotations):
            ann["id"] = i
        print(len(pseudo_images))
        mmcv.dump(
            {
                "annotations": pseudo_annotations,
                "images": pseudo_images,
                "categories": [{"supercategory": "wheat", "name": "wheat", "id": 1}],
            },
            output_path,
        )

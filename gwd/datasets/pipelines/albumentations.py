import inspect
import random
import sys

import albumentations as A
import cv2
import mmcv
import numpy as np
from albumentations.augmentations.bbox_utils import union_of_bboxes
from albumentations.augmentations.transforms import F

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Albu

from .utils import calculate_area, calculate_aspect_ratios


@F.preserve_channel_dim
def stretch(img, w_scale, h_scale, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    new_height, new_width = int(height * h_scale), int(width * w_scale)
    return F.resize(img, new_height, new_width, interpolation)


class RandomStretch(A.RandomScale):
    def get_params(self):
        return {
            "w_scale": random.uniform(self.scale_limit[0], self.scale_limit[1]),
            "h_scale": random.uniform(self.scale_limit[0], self.scale_limit[1]),
        }

    def apply(self, img, w_scale=1.0, h_scale=1.0, interpolation=cv2.INTER_LINEAR, **params):
        return stretch(img, w_scale, h_scale, interpolation)


class ModifiedShiftScaleRotate(A.ShiftScaleRotate):
    def get_params(self):
        self.params = super().get_params()
        return self.params


class RandomBBoxesSafeCrop(A.DualTransform):
    def __init__(self, num_rate=(0.1, 1.0), erosion_rate=0.0, min_edge_ratio=0.5, always_apply=False, p=1.0):
        super(RandomBBoxesSafeCrop, self).__init__(always_apply, p)
        self.erosion_rate = erosion_rate
        self.num_rate = num_rate
        self.min_edge_ratio = min_edge_ratio

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, **params):
        return F.random_crop(img, crop_height, crop_width, h_start, w_start)

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = img_h if erosive_h >= img_h else random.randint(erosive_h, img_h)
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "crop_height": crop_height,
                "crop_width": int(crop_height * img_w / img_h),
            }
        x, y, x2, y2 = union_of_bboxes(width=1.0, height=1.0, bboxes=params["bboxes"])
        if x2 - x >= self.min_edge_ratio and y2 - y >= self.min_edge_ratio:
            for i in range(50):
                # get union of all bboxes
                x, y, x2, y2 = union_of_bboxes(
                    width=1.0,
                    height=1.0,
                    bboxes=random.choices(
                        params["bboxes"], k=max(int(random.uniform(*self.num_rate) * len(params["bboxes"])), 1)
                    ),
                )
                # find bigger region
                bx, by = (
                    x * random.uniform(1 - self.erosion_rate, 1.0),
                    y * random.uniform(1 - self.erosion_rate, 1.0),
                )
                bx2, by2 = (
                    x2 + (1 - x2) * random.uniform(1 - self.erosion_rate, 1.0),
                    y2 + (1 - y2) * random.uniform(1 - self.erosion_rate, 1.0),
                )
                bw, bh = bx2 - bx, by2 - by
                crop_height = img_h if bh >= 1.0 else int(img_h * bh)
                crop_width = img_w if bw >= 1.0 else int(img_w * bw)

                if crop_height / crop_width < 0.5 or crop_height / crop_width > 2:
                    continue

                h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
                w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
                return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}
        return {"h_start": 0, "w_start": 0, "crop_height": img_h, "crop_width": img_w}

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return F.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return "erosion_rate", "num_rate", "min_edge_ratio"


def albu_builder(cfg):
    """Import a module from albumentations.
    Inherits some of `build_from_cfg` logic.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and "type" in cfg
    args = cfg.copy()

    obj_type = args.pop("type")
    if mmcv.is_str(obj_type):
        if A is None:
            raise RuntimeError("albumentations is not installed")
        if hasattr(A, obj_type):
            obj_cls = getattr(A, obj_type)
        else:
            obj_cls = getattr(sys.modules[__name__], obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

    if "transforms" in args:
        args["transforms"] = [albu_builder(transform) for transform in args["transforms"]]

    return obj_cls(**args)


@PIPELINES.register_module()
class Albumentations(Albu):
    def __init__(self, min_visibility=0.3, min_size=4, max_aspect_ratio=10, **kwargs):
        super(Albumentations, self).__init__(**kwargs)
        self.min_visibility = min_visibility
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio

        # Be careful: it is a dirty hack
        for i, t in enumerate(self.aug.transforms):
            if isinstance(t, ModifiedShiftScaleRotate):
                self.scale_index = i
        assert hasattr(self, "scale_index")

    def albu_builder(self, cfg):
        return albu_builder(cfg)

    def reset_scale_zero(self):
        self.aug.transforms[self.scale_index].params = {"scale": 1.0}

    def __call__(self, results):
        original_areas = calculate_area(results["gt_bboxes"])
        results = self.mapper(results, self.keymap_to_albu)
        if isinstance(results["bboxes"], np.ndarray):
            results["bboxes"] = [x for x in results["bboxes"]]
        results["labels"] = np.arange(len(results["bboxes"]))

        self.reset_scale_zero()
        results = self.aug(**results)
        scale = self.aug.transforms[self.scale_index].params["scale"]
        if isinstance(results["bboxes"], list):
            results["bboxes"] = np.array(results["bboxes"], dtype=np.float32)

        if not len(results["bboxes"]) and self.skip_img_without_anno:
            return None

        original_areas = original_areas[results["labels"]]
        augmented_areas = calculate_area(results["bboxes"])
        aspect_ratios = calculate_aspect_ratios(results["bboxes"])
        widths = results["bboxes"][:, 2] - results["bboxes"][:, 0]
        heights = results["bboxes"][:, 3] - results["bboxes"][:, 1]
        size_mask = (widths > self.min_size) & (heights > self.min_size)
        area_mask = augmented_areas / (scale ** 2 * original_areas) > self.min_visibility
        aspect_ratio_mask = aspect_ratios < self.max_aspect_ratio
        mask = size_mask & area_mask & aspect_ratio_mask
        results["bboxes"] = results["bboxes"][mask]
        if not len(results["bboxes"]) and self.skip_img_without_anno:
            return None
        results["gt_labels"] = np.zeros(len(results["bboxes"])).astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)
        # update final shape
        if self.update_pad_shape:
            results["pad_shape"] = results["img"].shape

        if results is not None:
            results["img_shape"] = results["img"].shape
        return results

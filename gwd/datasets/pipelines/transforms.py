import os.path as osp
import random
from glob import glob

import cv2
import mmcv
import numpy as np
from albumentations import Compose

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Pad

from .albumentations import albu_builder
from .utils import calculate_area

try:
    from bbaug import policies
except ImportError:
    policies = None


@PIPELINES.register_module()
class RandomRotate90(object):
    def __init__(self, rotate_ratio=None):
        self.rotate_ratio = rotate_ratio
        if rotate_ratio is not None:
            assert 0 <= rotate_ratio <= 1

    def bbox_rot90(self, bboxes, img_shape, factor):
        assert bboxes.shape[-1] % 4 == 0
        h, w = img_shape[:2]
        rotated = bboxes.copy()
        if factor == 1:
            rotated[..., 0] = bboxes[..., 1]
            rotated[..., 1] = w - bboxes[..., 2]
            rotated[..., 2] = bboxes[..., 3]
            rotated[..., 3] = w - bboxes[..., 0]
        elif factor == 2:
            rotated[..., 0] = w - bboxes[..., 2]
            rotated[..., 1] = h - bboxes[..., 3]
            rotated[..., 2] = w - bboxes[..., 0]
            rotated[..., 3] = h - bboxes[..., 1]
        elif factor == 3:
            rotated[..., 0] = h - bboxes[..., 3]
            rotated[..., 1] = bboxes[..., 0]
            rotated[..., 2] = h - bboxes[..., 1]
            rotated[..., 3] = bboxes[..., 2]
        return rotated

    def __call__(self, results):
        if "rotate" not in results:
            rotate = True if np.random.rand() < self.rotate_ratio else False
            results["rotate"] = rotate
        if "rotate_factor" not in results:
            rotate_factor = random.randint(0, 3)
            results["rotate_factor"] = rotate_factor
        if results["rotate"]:
            # rotate image
            for key in results.get("img_fields", ["img"]):
                results[key] = np.ascontiguousarray(np.rot90(results[key], results["rotate_factor"]))
            results["img_shape"] = results["img"].shape
            # rotate bboxes
            for key in results.get("bbox_fields", []):
                results[key] = self.bbox_rot90(results[key], results["img_shape"], results["rotate_factor"])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(factor={self.factor})"


@PIPELINES.register_module()
class RandomCropVisibility(object):
    def __init__(self, crop_size, min_visibility=0.5):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.min_visibility = min_visibility

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results["img_shape"] = img_shape

        if "gt_bboxes" in results:
            original_areas = calculate_area(results["gt_bboxes"])

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get("bbox_fields", []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
            bboxes = results[key] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

        # filter out the gt bboxes that are completely cropped
        if "gt_bboxes" in results:
            gt_bboxes = results["gt_bboxes"]
            gt_bboxes_ignore = results["gt_bboxes_ignore"]
            cropped_areas = calculate_area(gt_bboxes)
            valid_inds = (cropped_areas / original_areas) >= self.min_visibility
            ignore_valid_inds = (gt_bboxes_ignore[:, 2] > gt_bboxes_ignore[:, 0]) & (
                gt_bboxes_ignore[:, 3] > gt_bboxes_ignore[:, 1]
            )
            # if no gt bbox remains after cropping, just skip this image
            if not np.any(valid_inds):
                return None
            results["gt_bboxes"] = gt_bboxes[valid_inds, :]
            results["gt_bboxes_ignore"] = gt_bboxes_ignore[ignore_valid_inds, :]
            if "gt_labels" in results:
                results["gt_labels"] = results["gt_labels"][valid_inds]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


class BufferTransform(object):
    def __init__(self, min_buffer_size, p=0.5):
        self.p = p
        self.min_buffer_size = min_buffer_size
        self.buffer = []

    def apply(self, results):
        raise NotImplementedError

    def __call__(self, results):
        if len(self.buffer) < self.min_buffer_size:
            self.buffer.append(results.copy())
            return None
        if np.random.rand() <= self.p and len(self.buffer) >= self.min_buffer_size:
            random.shuffle(self.buffer)
            return self.apply(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nmin_buffer_size={self.min_buffer_size}),\n"
        repr_str += f"(\nratio={self.p})"
        return repr_str


@PIPELINES.register_module()
class Mosaic(BufferTransform):
    """
    Based on https://github.com/dereyly/mmdet_sota
    """

    def __init__(self, min_buffer_size=4, p=0.5, pad_val=0):
        assert min_buffer_size >= 4, "Buffer size for mosaic should be at least 4!"
        super(Mosaic, self).__init__(min_buffer_size=min_buffer_size, p=p)
        self.pad_val = pad_val

    def apply(self, results):
        # take four images
        a = self.buffer.pop()
        b = self.buffer.pop()
        c = self.buffer.pop()
        d = self.buffer.pop()
        # get max shape
        max_h = max(a["img"].shape[0], b["img"].shape[0], c["img"].shape[0], d["img"].shape[0])
        max_w = max(a["img"].shape[1], b["img"].shape[1], c["img"].shape[1], d["img"].shape[1])

        # cropping pipe
        padder = Pad(size=(max_h, max_w), pad_val=self.pad_val)

        # crop
        a, b, c, d = padder(a), padder(b), padder(c), padder(d)

        # check if cropping returns None => see above in the definition of RandomCrop
        if not a or not b or not c or not d:
            return results

        # offset bboxes in stacked image
        def offset_bbox(res_dict, x_offset, y_offset, keys=("gt_bboxes", "gt_bboxes_ignore")):
            for k in keys:
                if k in res_dict and res_dict[k].size > 0:
                    res_dict[k][:, 0::2] += x_offset
                    res_dict[k][:, 1::2] += y_offset
            return res_dict

        b = offset_bbox(b, max_w, 0)
        c = offset_bbox(c, 0, max_h)
        d = offset_bbox(d, max_w, max_h)

        # collect all the data into result
        top = np.concatenate([a["img"], b["img"]], axis=1)
        bottom = np.concatenate([c["img"], d["img"]], axis=1)
        results["img"] = np.concatenate([top, bottom], axis=0)
        results["img_shape"] = (max_h * 2, max_w * 2)

        for key in ["gt_labels", "gt_bboxes", "gt_labels_ignore", "gt_bboxes_ignore"]:
            if key in results:
                results[key] = np.concatenate([a[key], b[key], c[key], d[key]], axis=0)
        return results

    def __repr__(self):
        repr_str = self.__repr__()
        repr_str += f"(\npad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class Mixup(BufferTransform):
    def __init__(self, min_buffer_size=2, p=0.5, pad_val=0):
        assert min_buffer_size >= 2, "Buffer size for mosaic should be at least 2!"
        super(Mixup, self).__init__(min_buffer_size=min_buffer_size, p=p)
        self.pad_val = pad_val

    def apply(self, results):
        # take four images
        a = self.buffer.pop()
        b = self.buffer.pop()

        # get min shape
        max_h = max(a["img"].shape[0], b["img"].shape[0])
        max_w = max(a["img"].shape[1], b["img"].shape[1])

        # cropping pipe
        padder = Pad(size=(max_h, max_w), pad_val=self.pad_val)

        # crop
        a, b = padder(a), padder(b)

        # check if cropping returns None => see above in the definition of RandomCrop
        if not a or not b:
            return results

        # collect all the data into result
        results["img"] = ((a["img"].astype(np.float32) + b["img"].astype(np.float32)) / 2).astype(a["img"].dtype)
        results["img_shape"] = (max_h, max_w)

        for key in ["gt_labels", "gt_bboxes", "gt_labels_ignore", "gt_bboxes_ignore"]:
            if key in results:
                results[key] = np.concatenate([a[key], b[key]], axis=0)
        return results


@PIPELINES.register_module()
class RandomCopyPasteFromFile(object):
    def __init__(self, root, n_crops=3, transforms=(), crop_label=1, iou_threshold=0.5, p=0.5):
        self.root = root
        self.n_crops = n_crops
        self.crop_label = crop_label
        self.crop_paths = glob(osp.join(root, "*.jpg"))
        self.aug = Compose([albu_builder(t) for t in transforms])
        self.p = p
        self.iou_threshold = iou_threshold

    def __call__(self, results):
        if np.random.rand() <= self.p:
            for i in range(np.random.randint(1, self.n_crops + 1)):
                filepath = random.choice(self.crop_paths)
                crop = mmcv.imread(filepath)
                crop = self.aug(image=crop)["image"]
                crop_h, crop_w = crop.shape[:2]
                h, w = results["img"].shape[:2]
                if 3 <= crop_w < w and 3 <= crop_h < h:
                    for _ in range(10):
                        y_center = np.random.randint(crop_h // 2, h - crop_h // 2)
                        x_center = np.random.randint(crop_w // 2, w - crop_w // 2)
                        crop_bbox = np.array(
                            [
                                x_center - crop_w // 2,
                                y_center - crop_h // 2,
                                x_center + crop_w // 2,
                                y_center + crop_h // 2,
                            ]
                        ).reshape(1, 4)
                        ious = bbox_overlaps(results["gt_bboxes"], crop_bbox, mode="iof")
                        if max(ious) < self.iou_threshold:
                            crop_mask = 255 * np.ones(crop.shape, crop.dtype)
                            results["img"] = cv2.seamlessClone(
                                src=crop,
                                dst=results["img"],
                                mask=crop_mask,
                                p=(x_center, y_center),
                                flags=cv2.NORMAL_CLONE,
                            )
                            results["gt_bboxes"] = np.concatenate([results["gt_bboxes"], crop_bbox])
                            results["gt_labels"] = np.concatenate(
                                [results["gt_labels"], np.full(len(crop_bbox), self.crop_label)]
                            )
                            break
        return results

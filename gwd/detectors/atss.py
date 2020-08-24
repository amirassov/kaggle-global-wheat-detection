import torch

from mmdet.core import bbox2result  # , bbox_mapping_back
from mmdet.core.bbox import bbox_flip
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.ops.nms import batched_nms


def bbox_rot90_back(bboxes, img_shape, factor=0):
    assert bboxes.shape[-1] % 4 == 0
    assert factor in {0, 1, 2, 3}
    rotated = bboxes.clone()
    h, w = img_shape[:2]
    if factor == 3:
        rotated[..., 0] = bboxes[..., 1]
        rotated[..., 1] = w - bboxes[..., 2]
        rotated[..., 2] = bboxes[..., 3]
        rotated[..., 3] = w - bboxes[..., 0]
    elif factor == 2:
        rotated[..., 0] = w - bboxes[..., 2]
        rotated[..., 1] = h - bboxes[..., 3]
        rotated[..., 2] = w - bboxes[..., 0]
        rotated[..., 3] = h - bboxes[..., 1]
    elif factor == 1:
        rotated[..., 0] = h - bboxes[..., 3]
        rotated[..., 1] = bboxes[..., 0]
        rotated[..., 2] = h - bboxes[..., 1]
        rotated[..., 3] = bboxes[..., 2]
    return rotated


def bbox_mapping_back(
    bboxes,
    img_shape,
    scale_factor,
    flip,
    flip_direction="horizontal",
    # rotate=False,
    # rotate_factor=0,
):
    new_bboxes = bbox_flip(bboxes, img_shape, flip_direction) if flip else bboxes
    # new_bboxes = (
    #     bbox_rot90_back(new_bboxes, img_shape, rotate_factor) if rotate else new_bboxes
    # )
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None):
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]
    # filter out boxes with low scores
    scaled_scores = scores * score_factors[:, None]
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    scores = scores[valid_mask]
    scaled_scores = scaled_scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scaled_scores, labels, nms_cfg)
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    scores = scores[keep]
    dets[:, -1] = scores
    return dets, labels[keep]


@DETECTORS.register_module()
class ModifiedATSS(SingleStageDetector):
    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None):
        super(ModifiedATSS, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)

    def merge_aug_results(self, aug_bboxes, aug_scores, aug_centerness, img_metas):
        """Merge augmented detection bboxes and scores.
        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).
        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]["img_shape"]
            scale_factor = img_info[0]["scale_factor"]
            flip = img_info[0]["flip"]
            flip_direction = img_info[0]["flip_direction"]
            # rotate = img_info[0]["rotate"]
            # rotate_factor = img_info[0]["rotate_factor"]
            bboxes = bbox_mapping_back(
                bboxes=bboxes,
                img_shape=img_shape,
                scale_factor=scale_factor,
                flip=flip,
                flip_direction=flip_direction,
                # rotate=rotate,
                # rotate_factor=rotate_factor
            )
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        centerness = torch.cat(aug_centerness, dim=0)
        if aug_scores is None:
            return bboxes, centerness
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores, centerness

    def aug_test(self, imgs, img_metas, rescale=False):
        # recompute feats to save memory
        feats = self.extract_feats(imgs)

        aug_bboxes = []
        aug_scores = []
        aug_centerness = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            det_bboxes, det_scores, det_centerness = self.bbox_head.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(det_bboxes)
            aug_scores.append(det_scores)
            aug_centerness.append(det_centerness)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores, merged_centerness = self.merge_aug_results(
            aug_bboxes, aug_scores, aug_centerness, img_metas
        )
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            score_factors=merged_centerness,
        )

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(img_metas[0][0]["scale_factor"])
        bbox_results = bbox2result(_det_bboxes, det_labels, self.bbox_head.num_classes)
        return bbox_results

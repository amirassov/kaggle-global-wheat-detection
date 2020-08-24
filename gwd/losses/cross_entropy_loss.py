from functools import partial

import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses import CrossEntropyLoss, binary_cross_entropy, mask_cross_entropy
from mmdet.models.losses.utils import weight_reduce_loss


def label_smooth_cross_entropy(
    pred, label, weight=None, reduction="mean", avg_factor=None, class_weight=None, label_smooth=None
):
    # element-wise losses
    if label_smooth is None:
        loss = F.cross_entropy(pred, label, reduction="none")
    else:
        num_classes = pred.size(1)
        target = F.one_hot(label, num_classes).type_as(pred)
        target = target.sub_(label_smooth).clamp_(0).add_(label_smooth / num_classes)
        loss = F.kl_div(pred.log_softmax(1), target, reduction="none").sum(1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class LabelSmoothCrossEntropyLoss(CrossEntropyLoss):
    def __init__(
        self, use_sigmoid=False, use_mask=False, reduction="mean", loss_weight=1.0, class_weight=None, label_smooth=0.1
    ):
        super(LabelSmoothCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.label_smooth = label_smooth

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        elif self.label_smooth is None:
            self.cls_criterion = label_smooth_cross_entropy
        else:
            self.cls_criterion = partial(label_smooth_cross_entropy, label_smooth=self.label_smooth)

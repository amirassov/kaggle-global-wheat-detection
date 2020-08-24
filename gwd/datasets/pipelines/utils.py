import numpy as np


def calculate_area(bboxes):
    return (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])


def calculate_aspect_ratios(bboxes, eps=1e-6):
    return np.maximum(
        (bboxes[:, 2] - bboxes[:, 0]) / (bboxes[:, 3] - bboxes[:, 1] + eps),
        (bboxes[:, 3] - bboxes[:, 1]) / (bboxes[:, 2] - bboxes[:, 0] + eps),
    )

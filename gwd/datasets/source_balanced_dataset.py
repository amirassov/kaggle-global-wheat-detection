import math
from collections import defaultdict

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.dataset_wrappers import ClassBalancedDataset

from .wheat_detection import WheatDataset


@DATASETS.register_module()
class SourceBalancedDataset(ClassBalancedDataset):
    def _get_repeat_factors(self, dataset: WheatDataset, repeat_thr: float):
        # 1. For each source s, compute the fraction # of images
        #   that contain it: f(s)
        source_freq = defaultdict(float)
        num_images = len(dataset)
        for data_info in dataset.data_infos:
            source = data_info["source"]
            source_freq[source] += 1.0
        for k, v in source_freq.items():
            source_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        source_repeat = {
            source: max(1.0, math.sqrt(repeat_thr / source_freq)) for source, source_freq in source_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = [source_repeat[x["source"]] for x in dataset.data_infos]
        return repeat_factors

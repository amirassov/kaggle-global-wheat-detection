import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.loading import LoadImageFromFile


@PIPELINES.register_module()
class MultipleLoadImageFromFile(LoadImageFromFile):
    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        assert isinstance(results["img_prefix"], dict)
        img_prefix = np.random.choice(results["img_prefix"]["roots"], p=results["img_prefix"]["probabilities"])
        filename = osp.join(img_prefix, results["img_info"]["filename"])

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["img_fields"] = ["img"]
        return results

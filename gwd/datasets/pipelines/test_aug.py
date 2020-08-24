import warnings

import mmcv

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import MultiScaleFlipAug


@PIPELINES.register_module()
class ModifiedMultiScaleFlipAug(MultiScaleFlipAug):
    def __init__(self, rotate=False, rotate_factor=0, **kwargs):
        super(ModifiedMultiScaleFlipAug, self).__init__(**kwargs)
        self.rotate = rotate
        self.rotate_factor = rotate_factor if isinstance(rotate_factor, list) else [rotate_factor]
        assert mmcv.is_list_of(self.rotate_factor, int)
        if not self.rotate and self.rotate_factor != [0]:
            warnings.warn("rotate_factor has no effect when rotate is set to False")

    def __call__(self, results):
        aug_data = []
        flip_args = [[False, None]]
        rotate_args = [[False, None]]
        if self.flip:
            flip_args += [[True, direction] for direction in self.flip_direction]
        if self.rotate:
            rotate_args += [[True, factor] for factor in self.rotate_factor]
        for scale in self.img_scale:
            for flip, direction in flip_args:
                for rotate, factor in rotate_args:
                    _results = results.copy()
                    _results[self.scale_key] = scale
                    _results["flip"] = flip
                    _results["flip_direction"] = direction
                    _results["rotate"] = rotate
                    _results["rotate_factor"] = factor
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

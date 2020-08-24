_base_ = "./wheat_detection_mstrain_light.py"

data_root = "/data/"
data = dict(
    train=dict(
        ann_file=[
            data_root + "folds_v2/{fold}/coco_tile_train.json",
            data_root + "folds_v2/{fold}/coco_pseudo_train.json",
            data_root + "coco_spike.json",
        ],
        img_prefix=[
            dict(
                roots=[
                    data_root + "train/",
                    data_root + "colored_train/",
                    data_root + "stylized_train/",
                    data_root + "stylized_by_test_v1/",
                    data_root + "stylized_by_test_v2/",
                    data_root + "stylized_by_test_v3/",
                    data_root + "stylized_by_test_v4/",
                ],
                probabilities=[0.4, 0.3, 0.3 / 5, 0.3 / 5, 0.3 / 5, 0.3 / 5, 0.3 / 5],
            ),
            dict(
                roots=[
                    data_root + "crops_fold0/",
                    data_root + "colored_crops_fold0/",
                    data_root + "stylized_pseudo_by_test_v1/",
                    data_root + "stylized_pseudo_by_test_v2/",
                    data_root + "stylized_pseudo_by_test_v3/",
                    data_root + "stylized_pseudo_by_test_v4/",
                ],
                probabilities=[0.5, 0.3, 0.2 / 4, 0.2 / 4, 0.2 / 4, 0.2 / 4],
            ),
            dict(
                roots=[
                    data_root + "SPIKE_images/",
                    data_root + "stylized_SPIKE_images_v1/",
                    data_root + "stylized_SPIKE_images_v2/",
                    data_root + "stylized_SPIKE_images_v3/",
                    data_root + "stylized_SPIKE_images_v4/",
                ],
                probabilities=[0.7, 0.3 / 4, 0.3 / 4, 0.3 / 4, 0.3 / 4],
            ),
        ],
    )
)

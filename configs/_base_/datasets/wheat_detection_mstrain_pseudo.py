_base_ = "./wheat_detection_mstrain.py"

data_root = "/data/"
data = dict(
    samples_per_gpu=1,
    train=dict(
        ann_file=[
            data_root + "coco_train.json",
            data_root + "coco_pseudo_test.json",
            data_root + "coco_pseudo_test.json",
            data_root + "coco_pseudo_test.json",
        ],
        img_prefix=[data_root + "train/", data_root + "test/", data_root + "test/", data_root + "test/"],
    ),
)

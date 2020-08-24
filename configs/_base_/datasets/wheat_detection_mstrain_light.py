_base_ = "./wheat_detection_mstrain.py"

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(type="RandomRotate90", p=1.0),
    dict(
        type="OneOf",
        transforms=[
            dict(type="HueSaturationValue", hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),
            dict(type="RandomGamma"),
            dict(type="CLAHE"),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="RandomBrightnessContrast", brightness_limit=0.25, contrast_limit=0.25),
            dict(type="RGBShift", r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur"),
            dict(type="MotionBlur"),
            dict(type="GaussNoise"),
            dict(type="ImageCompression", quality_lower=75),
        ],
        p=0.4,
    ),
    dict(
        type="CoarseDropout",
        max_holes=30,
        max_height=30,
        max_width=30,
        min_holes=5,
        min_height=10,
        min_width=10,
        fill_value=img_norm_cfg["mean"][::-1],
        p=0.4,
    ),
    dict(
        type="ModifiedShiftScaleRotate",
        shift_limit=0.3,
        rotate_limit=5,
        scale_limit=(-0.3, 0.75),
        border_mode=0,
        value=img_norm_cfg["mean"][::-1],
    ),
    dict(type="RandomBBoxesSafeCrop", num_rate=(0.5, 1.0), erosion_rate=0.2),
]

train_pipeline = [
    dict(type="MultipleLoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Mosaic", p=0.25, min_buffer_size=4, pad_val=img_norm_cfg["mean"][::-1]),
    dict(
        type="Albumentations",
        transforms=albu_train_transforms,
        keymap=dict(img="image", gt_masks="masks", gt_bboxes="bboxes"),
        update_pad_shape=False,
        skip_img_without_anno=True,
        bbox_params=dict(type="BboxParams", format="pascal_voc", label_fields=["labels"]),
        min_visibility=0.3,
        min_size=4,
        max_aspect_ratio=15,
    ),
    dict(type="Mixup", p=0.25, min_buffer_size=2, pad_val=img_norm_cfg["mean"][::-1]),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="Resize",
        img_scale=[(768 + 32 * i, 768 + 32 * i) for i in range(25)],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

data_root = "/data/"
data = dict(
    train=dict(
        pipeline=train_pipeline,
        ann_file=[
            data_root + "folds_v2/{fold}/coco_tile_train.json",
            data_root + "folds_v2/{fold}/coco_pseudo_train.json",
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
                probabilities=[0.75, 0.15, 0.1 / 5, 0.1 / 5, 0.1 / 5, 0.1 / 5, 0.1 / 5],
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
                probabilities=[0.75, 0.15, 0.1 / 4, 0.1 / 4, 0.1 / 4, 0.1 / 4],
            ),
        ],
    )
)

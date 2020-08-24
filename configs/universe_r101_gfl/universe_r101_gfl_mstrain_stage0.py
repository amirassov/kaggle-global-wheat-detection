_base_ = [
    "../_base_/models/universe_r101_gfl.py",
    "../_base_/datasets/wheat_detection_mstrain_hard.py",
    "../_base_/schedules/schedule_4x.py",
    "../_base_/default_runtime.py",
]

data = dict(samples_per_gpu=6)
optimizer = dict(lr=0.03)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
load_from = "/dumps/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200716_epoch_24-1b9a1241.pth"
fp16 = dict(loss_scale=512.0)

_base_ = [
    "../_base_/models/universe_r101_gfl.py",
    "../_base_/datasets/wheat_detection_mstrain_pseudo.py",
    "../_base_/schedules/schedule_pseudo.py",
    "../_base_/default_runtime.py",
]

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
fp16 = dict(loss_scale=512.0)
checkpoint_config = dict(interval=1)

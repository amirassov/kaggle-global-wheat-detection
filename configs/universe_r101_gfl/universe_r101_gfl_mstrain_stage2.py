_base_ = [
    "../_base_/models/universe_r101_gfl.py",
    "../_base_/datasets/wheat_detection_mstrain.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

data = dict(samples_per_gpu=6)
optimizer = dict(lr=0.03 / 3)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
load_from = "/dumps/work_dirs/universe_r101_gfl_mstrain_stage1/0/epoch_48.pth"
fp16 = dict(loss_scale=512.0)
checkpoint_config = dict(interval=1)

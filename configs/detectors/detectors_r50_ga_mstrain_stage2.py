_base_ = [
    "../_base_/models/detectors_r50_ga.py",
    "../_base_/datasets/wheat_detection_mstrain.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

data = dict(samples_per_gpu=2)
optimizer = dict(lr=0.02 / 4)
load_from = "/dumps/work_dirs/detectors_r50_ga_mstrain_stage1/0/epoch_12.pth"
checkpoint_config = dict(interval=1)

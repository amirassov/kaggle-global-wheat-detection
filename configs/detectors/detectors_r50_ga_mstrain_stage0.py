_base_ = [
    "../_base_/models/detectors_r50_ga.py",
    "../_base_/datasets/wheat_detection_mstrain_hard.py",
    "../_base_/schedules/schedule_4x.py",
    "../_base_/default_runtime.py",
]

data = dict(samples_per_gpu=2)
optimizer = dict(lr=0.01)
load_from = "/dumps/DetectoRS_R50-0f1c8080_v2_attention.pth"

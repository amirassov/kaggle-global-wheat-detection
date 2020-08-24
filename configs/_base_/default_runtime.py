checkpoint_config = dict(interval=4)
# yapf:disable
log_config = dict(interval=10, hooks=[dict(type="TextLoggerHook")])
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

_base_ = "schedule_1x.py"

# optimizer
optimizer = dict(lr=0.01 / 2 / 4)

# learning policy
lr_config = dict(step=[1500, 3500], by_epoch=False)
total_epochs = 1

_base_ = "./detectors_r50_ga_mstrain_local_pseudo.py"

kaggle_working = "/kaggle/working/"
data_root = "/kaggle/input/global-wheat-detection/"
data = dict(train=dict(ann_file=kaggle_working + "coco_pseudo_test.json", img_prefix=data_root + "test/"))

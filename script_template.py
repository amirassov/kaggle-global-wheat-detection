import base64
import gzip
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Dict

from gwd import submit, test, train, wbf  # noqa E402
from gwd.converters import images2coco, kaggle2coco  # noqa E402

MODELS_ROOT = "/kaggle/input/gwd-models"
WHEELS_ROOT = "/kaggle/input/mmdetection-wheels"
INPUT_ROOT = "/kaggle/input/global-wheat-detection"
KAGGLE_WORKING = "/kaggle/working"

# this is base64 encoded source code
file_data: Dict = {file_data}

for path, encoded in file_data.items():
    path = Path(KAGGLE_WORKING) / path
    print(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system(f"export PYTHONPATH=${{PYTHONPATH}}:{KAGGLE_WORKING} && " + command)


run(f"python {KAGGLE_WORKING}/setup.py develop")
sys.path.append(KAGGLE_WORKING)

for wheel_name in [
    "addict-2.2.1-py3-none-any.whl",
    "mmcv-0.6.2-cp37-cp37m-linux_x86_64.whl",
    "terminal-0.4.0-py3-none-any.whl",
    "terminaltables-3.1.0-py3-none-any.whl",
    "pycocotools-12.0-cp37-cp37m-linux_x86_64.whl",
    "mmdet-2.2.0cdfc6a1-cp37-cp37m-linux_x86_64.whl",
    "ensemble_boxes-1.0.0-py3-none-any.whl",
]:
    wheel_path = osp.join(WHEELS_ROOT, wheel_name)
    run(f"pip install {wheel_path}")


CONFIG_DETECTORS = osp.join(KAGGLE_WORKING, "configs/detectors/detectors_r50_ga_mstrain_stage2.py")
CHECKPOINT_DETECTORS = osp.join(MODELS_ROOT, "detectors_r50_ga_mstrainv2_stage1_epoch_24.pth")
CHECKPOINT_UNIVERSE = osp.join(MODELS_ROOT, "universe_r101_gfl_mstrain_stage1_epoch_11.pth")
IMG_PREFIX = osp.join(INPUT_ROOT, "test")
ANN_FILE = "coco_test_images.json"
PSEUDO_PATH = f"{KAGGLE_WORKING}/coco_pseudo_test.json"
PSEUDO_SCORE_THRESHOLD = 0.75
PSEUDO_CONFIDENCE_THRESHOLD = 0.6
SUBMISSION_PATH = "submission.csv"

# make pseudo prediction
submit.main(
    img_prefix=IMG_PREFIX,
    configs=[CONFIG_DETECTORS],
    checkpoints=[CHECKPOINT_DETECTORS],
    submission_path=f"detectors_{SUBMISSION_PATH}",
    ann_file=ANN_FILE,
    pseudo_path=PSEUDO_PATH,
    pseudo_score_threshold=PSEUDO_SCORE_THRESHOLD,
    pseudo_confidence_threshold=PSEUDO_CONFIDENCE_THRESHOLD,
    flip=True,
    format_only=True,
    weights=None,
    iou_thr=None,
    score_thr=None,
)

# convert train samples to COCO
kaggle2coco.main(
    annotation_path=f"{INPUT_ROOT}/train.csv",
    output_path=f"{KAGGLE_WORKING}/coco_train.json",
    exclude_sources=["usask_1", "ethz_1"],
)

if len(os.listdir(IMG_PREFIX)) == 10:
    PSEUDO_CONFIG_DETECTORS = f"{KAGGLE_WORKING}/configs/detectors/detectors_r50_ga_mstrain_public_pseudo.py"
    PSEUDO_CONFIG_UNIVERSE = f"{KAGGLE_WORKING}/configs/universe_r101_gfl/universe_r101_gfl_mstrain_public_pseudo.py"
else:
    PSEUDO_CONFIG_DETECTORS = f"{KAGGLE_WORKING}/configs/detectors/detectors_r50_ga_mstrain_private_pseudo.py"
    PSEUDO_CONFIG_UNIVERSE = f"{KAGGLE_WORKING}/configs/universe_r101_gfl/universe_r101_gfl_mstrain_private_pseudo.py"

# retrain DetectoRS
run(
    f"python {KAGGLE_WORKING}/gwd/train.py "
    f"--config {PSEUDO_CONFIG_DETECTORS} "
    "--no-validate "
    f"--load-from {CHECKPOINT_DETECTORS} "
    f"--work-dir {KAGGLE_WORKING}/pseudo_detectors"
)
detectors_predictions, test_dataset = test.main(
    config=PSEUDO_CONFIG_DETECTORS,
    checkpoint=f"{KAGGLE_WORKING}/pseudo_detectors/latest.pth",
    img_prefix=IMG_PREFIX,
    ann_file=ANN_FILE,
    flip=True,
    format_only=True,
    options=dict(output_path=f"pseudo_detectors_{SUBMISSION_PATH}"),
)
test_dataset.pseudo_results(
    results=detectors_predictions,
    output_path=PSEUDO_PATH,
    pseudo_score_threshold=PSEUDO_SCORE_THRESHOLD,
    pseudo_confidence_threshold=PSEUDO_CONFIDENCE_THRESHOLD,
)

# retrain UniverseNet
run(
    f"python {KAGGLE_WORKING}/gwd/train.py "
    f"--config {PSEUDO_CONFIG_UNIVERSE} "
    "--no-validate "
    f"--load-from {CHECKPOINT_UNIVERSE} "
    f"--work-dir {KAGGLE_WORKING}/pseudo_universe"
)
universe_predictions, test_dataset = test.main(
    config=PSEUDO_CONFIG_UNIVERSE,
    checkpoint=f"{KAGGLE_WORKING}/pseudo_universe/latest.pth",
    img_prefix=IMG_PREFIX,
    ann_file=ANN_FILE,
    flip=True,
    format_only=True,
    options=dict(output_path=f"pseudo_universe_{SUBMISSION_PATH}"),
)
ensemble_predictions = wbf.main(
    [detectors_predictions, universe_predictions], weights=[0.65, 0.35], iou_thr=0.55, score_thr=0.45
)
test_dataset.format_results(results=ensemble_predictions, output_path=SUBMISSION_PATH)

#!/usr/bin/env bash

set -e

GPUS=8
PORT=${PORT:-29500}

CONFIG=configs/detectors/detectors_r50_ga_mstrain_stage2.py
CHECKPOINT=/dumps/PL_detectors_r50.pth

python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" \
    "$PROJECT_ROOT"/gwd/test.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out /data/crops_fold0_predictions.pkl \
    --ann-file /data/coco_crops_fold0.json \
    --img-prefix /data/crops_fold0

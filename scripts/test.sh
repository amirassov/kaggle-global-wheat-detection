#!/usr/bin/env bash

set -e

GPUS=8
PORT=${PORT:-29500}

CONFIG=configs/universe_r101_gfl_v2/universe_r101_gfl_mstrainv2_stage1.py
CHECKPOINT=/dumps/work_dirs/universe_r101_gfl_mstrainv2_stage1/0/epoch_23.pth

python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" \
    "$PROJECT_ROOT"/gwd/test.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out /data/universe_r101_gfl_mstrainv2_stage1_1024_crops_predictions.pkl \
    --eval bbox \
    --fold 0

#!/usr/bin/env bash

set -e

CONFIG=configs/detectors/detectors_r50_ga_mstrain_stage2.py

GPUS=$1
CHECKPOINT=/dumps/work_dirs/detectors_r50_ga_mstrainv2_stage1/0/detectors_r50_ga_mstrainv2_stage1_epoch_24.pth

for score_thr in 0.5
do
  for iou_thr in 0.5
  do
    python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$RANDOM" \
        "$PROJECT_ROOT"/gwd/test.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --eval bbox \
        --fold 0 \
        --score_thr $score_thr \
        --iou_thr $iou_thr \
        --log-file /dumps/thr_logs.log
  done
done

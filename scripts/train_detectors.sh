#!/usr/bin/env bash

set -e
bash scripts/train.sh configs/detectors/detectors_r50_ga_mstrain_stage0.py 4
wait
bash scripts/train.sh configs/detectors/detectors_r50_ga_mstrain_stage1.py 4
wait
bash scripts/train.sh configs/detectors/detectors_r50_ga_mstrain_stage2.py 4

#!/usr/bin/env bash

set -e
bash scripts/train.sh configs/universe_r101_gfl/universe_r101_gfl_mstrain_stage0.py 4
wait
bash scripts/train.sh configs/universe_r101_gfl/universe_r101_gfl_mstrain_stage1.py 4
wait
bash scripts/train.sh configs/universe_r101_gfl/universe_r101_gfl_mstrain_stage2.py 4

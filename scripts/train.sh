#!/usr/bin/env bash

set -e

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$RANDOM \
    "$PROJECT_ROOT"/gwd/train.py --config $CONFIG --launcher pytorch --fold 0 ${@:3}

#!/usr/bin/env bash

set -e

for folder in train crops_fold0 SPIKE_images
do
  for i in {1..4}
  do
    python "$PROJECT_ROOT"/gwd/stylize/run.py \
      --content-dir=/data/${folder} \
      --style-dir=/data/test \
      --output-dir=/data/stylized_${folder}_v${i}
  done
done

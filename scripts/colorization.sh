#!/usr/bin/env bash

set -e

for folder in train crops_fold0
do
  python "$PROJECT_ROOT"/gwd/colorization/generate.py \
    --img_pattern=/data/${folder}/*jpg \
    --weights_path=/dumps/pix2pix_gen.pth \
    --output_root=/data/colored_${folder}
done

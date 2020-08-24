set -e

for fold in /data/folds_v2/[0-9]*
do
  for mode in mosaic tile
  do
    for split in train val
    do
      echo "$fold"/"$mode"_"$split"
      python "$PROJECT_ROOT"/gwd/converters/kaggle2coco.py \
        --annotation_path="$fold"/"$mode"_"$split".csv \
        --output_path="$fold"/coco_"$mode"_"$split".json
      wait
    done
  done
done

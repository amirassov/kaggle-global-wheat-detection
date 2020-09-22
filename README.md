# :ear_of_rice: 9th Place Solution of [Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413252%2F8115a3b84299209abd11cd8e7167e31e%2FSelection_149.png?generation=1598252059641875&alt=media)

- Our team: [Miras Amir](https://www.linkedin.com/in/amirassov/), [Or Katz](https://www.linkedin.com/in/or-katz-9ba885114/), [Shlomo Kashani](https://www.linkedin.com/in/quantscientist)
- [Kaggle post](https://www.kaggle.com/c/global-wheat-detection/discussion/172569)
- Submission kernel: [pseudo ensemble: detectors (3 st)+universenet r10](https://www.kaggle.com/amiras/pseudo-ensemble-detectors-3-st-universenet-r10)

# Solution

## Summary
Our solution is based on the excellent [MMDetection framework](https://github.com/open-mmlab/mmdetection).
We trained an ensemble of the following models:
- [DetectoRS with the ResNet50 backbone](https://github.com/joe-siyuan-qiao/DetectoRS)
- [UniverseNet+GFL with the Res2Net101 backbone](https://github.com/shinya7y/UniverseNet)

To increase the score a single round of pseudo labelling was applied to each model. Additionally, for a much better generalization of our models, we used heavy augmentations.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F413252%2Fc2f34231c3dbffc0e8335b8d9cb15898%2FSelection_120.png?generation=1596619828315578&amp;alt=media)

## Jigsaw puzzles
In the original corpus provided by the organizers, the training images were cropped from an original set of larger images. Therefore, we collected and assembled the original puzzles resulting in a corpus of 1330 puzzle images. The puzzle collection algorithm we adopted was based on [this code](https://github.com/lRomul/argus-tgs-salt/blob/master/mosaic/create_mosaic.py). But we were unsuccessful in collecting the bounding boxes for puzzles. Mainly because of the existence of bounding boxes that are located on or in the vicinity the border of the image. For this reason, we generated crops for the puzzles offline in addition to training images and generated boxes for them using pseudo labelling.

## Validation approach
We used MultilabelStratifiedKFold with 5 folds of [iterative stratification](https://github.com/trent-b/iterative-stratification) stratified by the number of boxes, a median of box areas and source of images. We guaranteed that there isn’t any leak between the sub-folds, so that the images of one puzzle were used only in that one particular fold.

Referring to the [paper](https://arxiv.org/abs/2005.02162), one can see wheat heads from different sources. We assumed that the wheat heads of `usask_1, ethz_1` sources are very different from the test sources (`UTokyo_1, UTokyo_2, UQ_1, NAU_1`). Therefore, we did not use these sources for validation.

However, our validation scores did not correlate well with the Kaggle LB. We only noticed global improvements (for example, DetectoRS is better than UniverseNet). Local improvements such as augmentation parameters, WBF parameters etc. did not correlate. We, therefore, shifted our attention to the LB scores mainly.

We trained our models only on the first fold.

## Augmentations
Due to the relatively small size of our training set, and another test set distribution, our approach relied heavily on data augmentation. During training, we utilized an extensive data augmentation protocol:
- Various augmentations from [albumentations](https://albumentations.ai):
    - HorizontalFlip, ShiftScaleRotate, RandomRotate90
    - RandomBrightnessContrast, HueSaturationValue, RGBShift
    - RandomGamma
    - CLAHE
    - Blur, MotionBlur
    - GaussNoise
    - ImageCompression
    - CoarseDropout
- RandomBBoxesSafeCrop. Randomly select N boxes in the image and find their union. Then we cropped the image keeping this unified.
- [Image colorization](https://www.kaggle.com/orkatz2/pytorch-pix-2-pix-for-image-colorization)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413252%2Fe469741f86a27bad9e9f23c22fb758f1%2Fcolored.jpg?generation=1598080876141360&alt=media)
- [Style transfer](https://github.com/bethgelab/stylize-datasets). A random image from a small test (10 images) was used as a style.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413252%2F66df1a63023a5636e960f0a9b04c4850%2FBeFunky-collage.jpg?generation=1597411461053544&alt=media)
- Mosaic augmentation. `a, b, c, d` -- randomly selected images. Then we just do the following:
```
top = np.concatenate([a, b], axis=1)
bottom = np.concatenate([c, d], axis=1)
result = np.concatenate([top, bottom], axis=0)
```
- Mixup augmentation. `a, b` -- randomly selected images. Then: `result = (a + b) / 2`
- Multi-scale Training. In each iteration, the scale of image is randomly sampled from `[(768 + 32 * i, 768 + 32 * i) for i in range(25)]`.
- All augmentations except colorization and style transfer were applied online.
Examples of augmented images:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F413252%2F54b1605a052bc34520cce5f7ca81f86f%2F0.jpg?generation=1596619282971124&amp;alt=media)   | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F413252%2Fe7bad4c91782a5614ee5731f3610e376%2F6.jpg?generation=1596619343754665&amp;alt=media)
:-------------------------:|:-------------------------:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F413252%2F4dfab25fb6ed84e7ef545c42361382c5%2F27.jpg?generation=1596619635969807&amp;alt=media) | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F413252%2Fcb53ccda22bf16bef104caf9e25459df%2F47.jpg?generation=1596619666130434&amp;alt=media)

## External data:
[SPIKE dataset](https://www.kaggle.com/c/global-wheat-detection/discussion/164346)

## Models
We used DetectoRS with ResNet50 and UniverseNet+GFL with Res2Net101 as main models. DetectoRS was a little bit more accurate and however much slower to train than UniverseNet:
- Single DetectoRS Public LB score without pseudo labeling: 0.7592
- Single UniverseNet Public LB score without pseudo labeling: 0.7567

For DetectoRS we used:
- LabelSmoothCrossEntropyLoss with parameter `0.1`
- [Empirical Attention](https://github.com/open-mmlab/mmdetection/tree/master/configs/empirical_attention)

## Training pipeline
In general, we used a multi-stage training pipeline:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F413252%2F6b28b3ab6458d9a0763d34c78abf15a0%2FSelection_125.png?generation=1596629231468089&amp;alt=media)

## Model inference
We used TTA6 (Test Time Augmentation) for all our models:
- Multi-scale Testing with scales `[(1408, 1408), (1536, 1536)]`
- Flips: `[original, horizontal, vertical]`

For TTA was used a standard MMDet algorithm with NMS that looks like this for two-stage detectors (DetectoRS):
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413252%2F6fff06749b880222e0c06bea777b8e84%2FSelection_126.png?generation=1597392868005836&alt=media)

For one-stage detectors (UniverseNet), the algorithm is similar, only without the part with RoiAlign, Head, etc.

## Pseudo labelling
- Sampling positive examples. We predicted the test image and received its scores and the bounding boxes. Then we calculated `confidence = np.mean(scores > 0.75)`. If the confidence was greater than 0.6 we accepted this image and used for pseudo labelling.
- Sources `[usask_1, ethz_1]` and augmentations like mosaic, mixup, colorization, style transfer weren’t used for pseudo labelling.
- 1 epoch, 1 round, 1 stage.
- Data: original data + pseudo test data :heavy_multiplication_x: 3

## Ensemble
We used [WBF](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) for the ensemble. The distribution of DetectoRS and UniverseNet scores is different. So we applied scaling using [rankdata](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html):
```
scaled_scores = 0.5 * (rankdata(scores) / len(scores)) + 0.5.
```

WBF parameters:
- `weights=[0.65, 0.35]` respectively for models `[DetectoRS, UniverseNet]`
- `iou_thr=0.55`
- `score_thr=0.45`

## Some observations from our submissions:
- Final submission: 0.6741 on Private LB and 0.7725 on Public LB
- Pseudo crops from jigsaw puzzles (DetectoRS R50): 0.7513 -&gt; 0.7582
- Tuning of pseudo labeling parameters for sampling positive examples (ensemble): 0.7709 -&gt; 0.7729
- Pseudo labeling (DetectoRS R50): 0.7582 -&gt; 0.7691
- Pseudo labeling (UniverseNet Res2Net50): 0.7494 -&gt; 0.7627
- SPIKE dataset (DetectoRS R50): 0.7582 -&gt; 0.7592
- Deleting [usask1, ethz1] from pseudo labeling (DetectoRS R50): 0.7678 -&gt; 0.7691

# How to run
## Data structure
```
/data/
├── train/
│   ├── d47799d91.jpg
│   ├── b57bb71b6.jpg
│   └── ...
├── train.csv
/dumps/
├── decoder.pth # checkpoint for style transfer (https://yadi.sk/d/begkgtQHxLo6kA)
├── vgg_normalised.pth # checkpoint for style transfer (https://yadi.sk/d/4BkKpSZ-4PUHqQ)
├── pix2pix_gen.pth # checkpoint for image colorization (https://yadi.sk/d/E5vAckDoFbuWYA)
└── PL_detectors_r50.pth # checkpoint of DetectoRS, which was trained without pseudo crops from jigsaw puzzles (https://yadi.sk/d/vpy2oXHFKGuMOg).
```

## Collecting jigsaw puzzles
```
python gwd/jigsaw/calculate_distance.py
python gwd/jigsaw/collect_images.py
python gwd/jigsaw/collect_bboxes.py
```

## Preparing folds
```
python gwd/split_folds.py
bash scripts/kaggle2coco.sh
```

## Pseudo crops from jigsaw puzzles
```
python gwd/jigsaw/crop.py
bash scripts/test_crops.sh
python gwd/prepare_pseudo.py
```

## Preparing external data
```
python gwd/converters/spike2kaggle.py
```

## Colorization and style transfer
```
bash scripts/colorization.sh
bash scripts/stylize.sh
```

## Training
```
bash scripts/train_detectors.sh
bash scripts/train_universenet.sh
```

## Submission preparing
It is based on [lopuhin/kaggle-script-template](https://github.com/lopuhin/kaggle-script-template).

You must upload the best checkpoints of trained models to the kaggle dataset. Further change the variables in `script_template.py`:
- `MODELS_ROOT` (name of your kaggle dataset)
- `CHECKPOINT_DETECTORS` (the best checkpoint of DetectoRS)
- `CHECKPOINT_UNIVERSE` (the best checkpoint of UniverseNet)

```
bash build.sh  # create a submission file ./build/script.py
```

Link to the kaggle dataset with:
- pretrained weights: [gwd_models](https://www.kaggle.com/amiras/gwd-models)
- external libraries: [mmdetection_wheels](https://www.kaggle.com/amiras/mmdetection-wheels)


## References
1. https://github.com/open-mmlab/mmdetection
2. https://github.com/shinya7y/UniverseNet
3. https://github.com/lopuhin/kaggle-script-template
4. https://github.com/trent-b/iterative-stratification
5. https://github.com/albumentations-team/albumentations
6. https://github.com/dereyly/mmdet_sota
7. https://github.com/eriklindernoren/PyTorch-GAN
8. https://github.com/bethgelab/stylize-datasets

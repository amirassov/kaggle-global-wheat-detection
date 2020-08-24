import warnings  # noqa

warnings.simplefilter("ignore", UserWarning)  # noqa

from .assigners.atss_assigner import FixedATSSAssigner
from .backbones.res2net import FixedRes2Net
from .backbones.sac.conv_aws import ConvAWS2d
from .backbones.sac.resnet import DetectoRS_ResNet
from .backbones.sac.resnext import DetectoRS_ResNeXt
from .backbones.sac.saconv import SAConv2d
from .datasets.pipelines.albumentations import Albumentations, ModifiedShiftScaleRotate, RandomBBoxesSafeCrop
from .datasets.pipelines.loading import MultipleLoadImageFromFile
from .datasets.pipelines.test_aug import ModifiedMultiScaleFlipAug
from .datasets.pipelines.transforms import (
    Mixup,
    Mosaic,
    RandomCopyPasteFromFile,
    RandomCropVisibility,
    RandomRotate90,
)
from .datasets.source_balanced_dataset import SourceBalancedDataset
from .datasets.wheat_detection import WheatDataset
from .dense_heads.gfl_head import GFLSEPCHead
from .detectors.atss import ModifiedATSS
from .detectors.rfp import RecursiveFeaturePyramid
from .losses.cross_entropy_loss import LabelSmoothCrossEntropyLoss
from .necks.sepc import SEPC
from .patches import build_dataset

import numpy as np
import torch

from gwd.eda.kmeans import kmeans
from mmdet.core.anchor import AnchorGenerator, build_anchor_generator


def main():
    anchor_generator_cfg = dict(type="AnchorGenerator", scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64])
    anchor_generator: AnchorGenerator = build_anchor_generator(anchor_generator_cfg)
    multi_level_anchors = anchor_generator.grid_anchors(
        featmap_sizes=[
            torch.Size([256, 256]),
            torch.Size([128, 128]),
            torch.Size([64, 64]),
            torch.Size([32, 32]),
            torch.Size([16, 16]),
        ],
        device="cpu",
    )
    anchors = torch.cat(multi_level_anchors).numpy()
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    data = np.stack([heights, widths], axis=1)
    clusters = kmeans(data, k=50)
    print(f"aspect rations: {clusters[: 0] / clusters[: 1]}")
    print(f"sizes: {np.sqrt(clusters[: 0] * clusters[: 1])}")


if __name__ == "__main__":
    main()

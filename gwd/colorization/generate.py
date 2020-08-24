import argparse
import os
import os.path as osp
from glob import glob

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm

from gwd.colorization.models import GeneratorUNet

IMG_SIZE = 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_pattern", default="/data/SPIKE_Dataset/images/*jpg")
    parser.add_argument("--weights_path", default="/dumps/pix2pix_gen.pth")
    parser.add_argument("--output_root", default="/data/SPIKE_Dataset/colored_images")
    return parser.parse_args()


def generate(model, img_path, transform):
    image = pil_loader(img_path)
    image = transform(image)
    with torch.no_grad():
        fake_image = model(image.unsqueeze(0).cuda())
    fake_image = fake_image.cpu().numpy()[0].transpose(1, 2, 0)
    fake_image -= fake_image.min()
    fake_image /= fake_image.max()
    fake_image *= 255
    fake_image = fake_image.astype(np.uint8)
    fake_image = cv2.UMat(fake_image).get()
    return fake_image


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    img_paths = glob(args.img_pattern)
    model = GeneratorUNet().cuda()
    model.load_state_dict(torch.load(args.weights_path, map_location="cpu"))
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    for img_path in tqdm(img_paths):
        fake_image = generate(model, img_path, transform)
        output_path = osp.join(args.output_root, osp.basename(img_path))
        cv2.imwrite(output_path, fake_image[..., ::-1])


if __name__ == "__main__":
    main()

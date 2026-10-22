"""
Packs CIFAR-10 into two .npz files (`x`: uint8 (N, 32, 32, 3), `y`: labels) for test/train.py and test/evaluate.py.

    python test/prepare_cifar10.py <folder with train/<class>/*.jpg and test/<class>/*.jpg> <output folder>

The image folders can come from any CIFAR-10 export, for example https://github.com/YoongiKim/CIFAR-10-images.
"""

import os
import sys

import numpy as np
from PIL import Image


def main(root: str, out: str):
    os.makedirs(out, exist_ok=True)
    for split in ("train", "test"):
        classes = sorted(os.listdir(os.path.join(root, split)))
        xs, ys = [], []
        for ci, c in enumerate(classes):
            folder = os.path.join(root, split, c)
            for f in sorted(os.listdir(folder)):
                xs.append(np.asarray(Image.open(os.path.join(folder, f)).convert("RGB")))
                ys.append(ci)
        np.savez(os.path.join(out, f"cifar10_{split}.npz"), x=np.stack(xs), y=np.array(ys, dtype=np.int64), classes=np.array(classes))
        print(split, len(xs), classes)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

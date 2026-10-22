#!/usr/bin/env bash
# Prepares the CIFAR-10 bench used by test/train.py and test/evaluate.py on a fresh machine: Python packages, the data
# (packed into .npz files) and the InceptionV3 weights for FID.
#
#     bash test/setup_bench.sh [data folder, default /home/user/data]
set -euo pipefail
DATA=${1:-/home/user/data}
HERE=$(cd "$(dirname "$0")" && pwd)
pip install -q torch torchvision transformers pytorch-fid scipy pillow numpy pytest
mkdir -p "$DATA"
if [ ! -f "$DATA/cifar10_train.npz" ]; then
  [ -d "$DATA/cifar10_jpg" ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images "$DATA/cifar10_jpg"
  python3 "$HERE/prepare_cifar10.py" "$DATA/cifar10_jpg" "$DATA"
fi
WEIGHTS=~/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth
if [ ! -f "$WEIGHTS" ]; then
  mkdir -p "$(dirname "$WEIGHTS")"
  curl -sSL -o "$WEIGHTS" https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth
fi
echo "bench ready in $DATA"

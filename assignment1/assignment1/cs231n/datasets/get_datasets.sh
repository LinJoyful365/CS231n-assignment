#!/bin/bash
# 下载 CIFAR-10 数据集到当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR"

if [ ! -d "$DATASET_DIR/cifar-10-batches-py" ]; then
  wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O "$DATASET_DIR/cifar-10-python.tar.gz"
  tar -xzvf "$DATASET_DIR/cifar-10-python.tar.gz" -C "$DATASET_DIR"
  rm "$DATASET_DIR/cifar-10-python.tar.gz"
  wget http://cs231n.stanford.edu/imagenet_val_25.npz -P "$DATASET_DIR"
fi

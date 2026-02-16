#!/bin/bash
# 下载 CIFAR-10 数据集到指定目录
DATASET_DIR="/Volumes/WorkDrive/CS231n/assignment1_colab/assignment1/cs231n/datasets"

if [ ! -d "$DATASET_DIR/cifar-10-batches-py" ]; then
  wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O "$DATASET_DIR/cifar-10-python.tar.gz"
  tar -xzvf "$DATASET_DIR/cifar-10-python.tar.gz" -C "$DATASET_DIR"
  rm "$DATASET_DIR/cifar-10-python.tar.gz"
  wget http://cs231n.stanford.edu/imagenet_val_25.npz -P "$DATASET_DIR"
fi
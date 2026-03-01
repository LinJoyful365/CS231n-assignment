# CS231n Assignment 实现记录（Local / Non-Colab）

本仓库用于记录笔者对 Stanford CS231n 课程作业（Assignment 1/2/3）的实现过程与实验结果。  
全程在本地环境完成，没有使用 Google Colab。  
后续将持续补充整理后的学习笔记与复盘内容。

## 仓库结构

```text
CS231n/
├── assignment1/
│   └── assignment1/
├── assignment2/
│   └── assignment2/
├── assignment3/
│   └── assignment3/
├── cs231n_blog.pdf
└── images/
```

说明：
- `assignment1/assignment1`：A1（KNN、SVM/Softmax、两层网络、全连接网络、特征工程）
- `assignment2/assignment2`：A2（BatchNorm、Dropout、CNN、PyTorch、RNN Captioning）
- `assignment3/assignment3`：A3（Transformer Captioning、Self-Supervised Learning、DDPM、CLIP/DINO）
- `cs231n_blog.pdf`：课程笔记 PDF（从线性分类器到扩散模型）
- `images/`：笔记中使用的 slides / 插图资源

## 目录命名说明

已将历史目录中的 `*_colab` 后缀移除，统一为：
- `assignment1`
- `assignment2`
- `assignment3`

注：部分 notebook 元数据中仍可能保留 `colab` 字段，这是原始模板遗留信息，不影响本地运行。

## 环境准备（建议）

### 通用

1. 安装 Python（建议 3.9/3.10）与 `pip`。
2. 建议使用 `conda` 或 `venv` 创建独立环境。
3. 在每个 assignment 目录单独安装依赖并运行 notebook。

## Assignment 1

工作目录：

```bash
cd assignment1/assignment1
```

数据下载：

```bash
bash cs231n/datasets/get_datasets.sh
```

建议按 notebook 顺序运行：
- `knn.ipynb`
- `softmax.ipynb`
- `two_layer_net.ipynb`
- `features.ipynb`
- `FullyConnectedNets.ipynb`

打包提交（代码 + PDF）：

```bash
bash collectSubmission.sh
```

## Assignment 2

工作目录：

```bash
cd assignment2/assignment2
```

数据下载：

```bash
bash cs231n/datasets/get_datasets.sh
bash cs231n/datasets/get_coco_dataset.sh
```

如需编译 Cython（卷积加速相关）：

```bash
cd cs231n
python setup.py build_ext --inplace
cd ..
```

建议按 notebook 顺序运行：
- `BatchNormalization.ipynb`
- `Dropout.ipynb`
- `ConvolutionalNetworks.ipynb`
- `PyTorch.ipynb`
- `RNN_Captioning_pytorch.ipynb`

打包提交（代码 + PDF）：

```bash
bash collectSubmission.sh
```

## Assignment 3

工作目录：

```bash
cd assignment3/assignment3
```

安装依赖（仓库内提供 requirements）：

```bash
pip install -r requirements.txt
```

数据下载（按需）：

```bash
bash cs231n/datasets/get_datasets.sh
bash cs231n/datasets/get_coco_dataset.sh
```

建议按 notebook 顺序运行：
- `Transformer_Captioning.ipynb`
- `Self_Supervised_Learning.ipynb`
- `DDPM.ipynb`
- `CLIP_DINO.ipynb`

打包提交（代码 + PDF）：

```bash
bash collectSubmission.sh
```

## 笔记与复盘

当前已提交：
- `cs231n_blog.pdf`（完整课程笔记）
- `images/CNN.png`
- `images/ReLUvsGELU.png`
- `images/cnn.jpeg`
- `images/ddpm.png`
- `images/im2col.png`
- `images/transformer.png`
- `images/vit.jpg`

## 其他说明

- `.gitignore` 已配置数据集与大文件忽略规则，避免提交超大实验产物。
- 如 notebook 中存在绝对路径，请按本机实际路径修改对应变量（通常在前几格）。
- 本仓库的重点是“可复现的本地作业实现记录”，不是 Colab 教程。

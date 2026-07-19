# Rethinkfun-DL

这是一个基于 rethinkfun 深度学习教程 fork 后维护的个人学习项目，用于系统练习 Deep Learning 与 PyTorch。仓库内容以章节代码为主，配合本地 WSL2 + Conda + NVIDIA GPU 环境，用来做从基础张量运算到卷积网络、迁移学习、Transformer 和机器翻译的代码实操。

## 项目定位

本项目不是一个单一应用或可发布库，而是一个面向学习的代码实验仓库：

- 通过短脚本理解 Tensor、Autograd、Loss、Optimizer 和训练循环；
- 通过章节代码复现实战案例，如 Titanic、MNIST、猫狗分类、CNN、ResNet、UNet 和 Transformer；
- 通过可验证的 PyTorch 环境保证 GPU、DataLoader、torchvision ops、TensorBoard 等学习工具可用；
- 保存个人学习过程中的环境搭建说明、工具选择建议和验证报告。

## 目录结构

```text
.
├── chapter5/     # 线性回归基础
├── chapter6/     # Autograd、Normalization、Loss 可视化、PyTorch 线性回归
├── chapter7/     # Titanic 传统数据建模练习
├── chapter8/     # MNIST、Titanic 与 PyTorch 入门训练
├── chapter10/    # 图像增强与猫狗分类
├── chapter11/    # LeNet、AlexNet、ResNet、UNet、迁移学习
├── chapter14/    # 分词器、翻译推理与 BLEU
├── chapter15/    # Transformer 训练、推理与 BLEU
├── Codex_WSL2_PyTorch环境自动化搭建与验证指南.md
├── PyTorch_AI代码实操分层学习指南.md
└── PyTorch_torch环境搭建验证报告与手动sudo检查.md
```

## 推荐环境

推荐使用项目中已整理的 `torch` Conda 环境：

```bash
conda activate torch
```

当前已验证环境：

| 项目 | 版本/状态 |
|---|---|
| Python | `3.11.15` |
| PyTorch | `2.12.1+cu130` |
| torchvision | `0.27.1+cu130` |
| PyTorch CUDA Runtime | `13.0` |
| GPU | `NVIDIA GeForce RTX 5080` |
| Jupyter Kernel | `Python (torch)` |

环境搭建、验证和手动 `sudo` 检查见：

- `Codex_WSL2_PyTorch环境自动化搭建与验证指南.md`
- `PyTorch_torch环境搭建验证报告与手动sudo检查.md`

## 快速验证

确认 PyTorch 和 GPU：

```bash
conda activate torch
python - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

运行完整环境验证：

```bash
conda activate torch
python ~/workspace/pytorch-playground/verify_torch_env.py
python ~/workspace/pytorch-playground/verify_fashion_mnist.py
python -m pip check
```

## 常用学习工具

| 场景 | 推荐工具 |
|---|---|
| 几行 Tensor/API 验证 | `ipython` |
| 一行环境检查 | `python -c` |
| 多行临时验证 | shell here-document |
| 终端连接 Jupyter kernel | `jupyter console --kernel torch` |
| 单函数或单算子学习 | 小型 `.py` + IPython `%run` |
| `nn.Module` 和梯度实验 | `.py` + `pytest` + `torchinfo` |
| 数据探索和可视化 | JupyterLab / Notebook |
| 最小训练闭环 | 训练脚本 + TensorBoard |
| 长时间训练 | `tmux` + 日志 + `nvitop` |

更多分层建议见 `PyTorch_AI代码实操分层学习指南.md`。

## 数据与大文件

仓库中的小型示例数据可以保留；大型原始数据集、训练输出和中间产物不应直接提交。

- `PetImages/` 是猫狗分类练习的大型图片数据目录，已在 `.gitignore` 中专门忽略；
- `checkpoints/`、`runs/`、`outputs/`、`wandb/` 等训练产物目录会被忽略；
- `.gitattributes` 已配置 `*.pt` 和 `*.zip` 通过 Git LFS 管理，适合确实需要版本化的大文件。

## 运行示例

```bash
conda activate torch
python chapter6/PyTorchLinearRegression.py
python chapter8/MNIST_pytorch.py
python chapter11/TransferLearning.py
```

部分脚本中的数据路径可能来自原教程或个人机器，需要按本地目录调整。高频读取的数据建议放在 WSL Linux 文件系统中，例如 `~/datasets`，不要放在 `/mnt/c` 或 `/mnt/d` 下。

## 维护原则

- 学习代码优先保持短小、可运行、可修改；
- 需要复现实验时，把训练入口、配置、日志和 checkpoint 分开管理；
- 不把 Notebook 当作唯一训练入口；
- 不在 WSL2 中安装 NVIDIA Linux 显示驱动；
- PyTorch 核心包优先使用官方 CUDA wheel，避免混装 Conda PyTorch 与 pip PyTorch。

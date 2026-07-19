# PyTorch `torch` Conda 环境搭建验证报告与手动 sudo 检查

生成时间：2026-07-18  
项目目录：`/home/cyrus/workspace/Rethinkfun-DL`  
目标环境：Conda 环境 `torch`

## 1. 环境结果

`torch` 环境已创建并验证通过。

| 项目 | 结果 |
|---|---|
| Conda 环境 | `torch` |
| 环境路径 | `/home/cyrus/anaconda3/envs/torch` |
| Python | `3.11.15` |
| PyTorch | `2.12.1+cu130` |
| torchvision | `0.27.1+cu130` |
| PyTorch CUDA Runtime | `13.0` |
| GPU | `NVIDIA GeForce RTX 5080` |
| Jupyter Kernel | `Python (torch)` |
| 依赖检查 | `python -m pip check` 通过 |

进入环境：

```bash
conda activate torch
```

## 2. 已完成验证

离线核心验证已通过：

- PyTorch 与 torchvision 版本检查；
- CUDA 可用性检查；
- GPU 设备名检查；
- Autograd 前向、反向与梯度有限性检查；
- `nn.Module` 前向、Loss 与反向传播检查；
- torchvision CUDA NMS 算子检查；
- BF16 autocast 检查；
- DataLoader 到 GPU 的 batch 传输检查。

成功标志：

```text
ALL CORE CHECKS PASSED
```

联网数据集验证已通过：

- Fashion-MNIST 下载；
- DataLoader 读取真实 batch；
- 图像和标签 shape、dtype、数值范围检查；
- GPU 上执行一次 forward、loss、backward、optimizer step。

成功标志：

```text
FASHION-MNIST SMOKE TEST PASSED
```

验证脚本位置：

```text
/home/cyrus/workspace/pytorch-playground/verify_torch_env.py
/home/cyrus/workspace/pytorch-playground/verify_fashion_mnist.py
```

日志与环境快照：

```text
/home/cyrus/setup-logs/torch-env/preflight.txt
/home/cyrus/setup-logs/torch-env/install.txt
/home/cyrus/setup-logs/torch-env/verify.txt
/home/cyrus/setup-logs/torch-env/freeze.txt
/home/cyrus/setup-logs/torch-env/requirements-lock.txt
/home/cyrus/setup-logs/torch-env/environment.yml
/home/cyrus/setup-logs/torch-env/environment-minimal.yml
```

## 3. 已安装主要包与工具

| 包/工具 | 功能 |
|---|---|
| `torch` | PyTorch 深度学习核心框架。 |
| `torchvision` | 视觉数据集、图像变换、模型和 CUDA vision ops。 |
| `cuda-toolkit` / `nvidia-*` wheel | PyTorch wheel 自带的 CUDA runtime、cuDNN、cuBLAS、NCCL 等 GPU 后端库。 |
| `triton` | PyTorch 编译和高性能 kernel 相关组件。 |
| `ipython` | 日常 Tensor/API 即时验证的首选交互环境。 |
| `jupyterlab` | Notebook、代码、终端和可视化整合环境。 |
| `jupyter-console` | 在终端里连接 Jupyter kernel 的轻量交互方式。 |
| `ipykernel` | 为 Jupyter 注册并运行 Python kernel。 |
| `ipywidgets` | Jupyter 交互控件支持。 |
| `numpy` | 数组计算和 PyTorch 张量概念对照学习。 |
| `scipy` | 科学计算、优化、线性代数和信号处理。 |
| `pandas` | 表格数据读取、清洗和分析。 |
| `scikit-learn` | 传统机器学习、数据划分和指标工具。 |
| `matplotlib` | 绘图、训练曲线和数据可视化。 |
| `pillow` | 图像读写和基础图像处理。 |
| `tqdm` | 训练循环和数据处理进度条。 |
| `rich` | 更友好的终端输出和日志显示。 |
| `pyyaml` | YAML 配置文件读取。 |
| `psutil` | CPU、内存和进程状态读取。 |
| `tensorboard` | 训练曲线、指标、图像和日志可视化。 |
| `torchinfo` | 查看模型层结构、输出形状和参数量。 |
| `torchmetrics` | 标准化训练/验证指标计算。 |
| `einops` | 清晰表达 Tensor 维度重排。 |
| `torchviz` | 小型计算图可视化。 |
| `graphviz` / `dot` | 为 `torchviz` 等工具渲染计算图。 |
| `timm` | 常用视觉模型和预训练模型库。 |
| `opencv-python-headless` | 无 GUI 版 OpenCV，适合 WSL/服务器图像处理。 |
| `albumentations` | 图像增强流水线。 |
| `scikit-image` | 传统图像处理和图像分析。 |
| `h5py` | HDF5 数据格式读写。 |
| `pyarrow` | Arrow/Parquet 等列式数据格式支持。 |
| `pytest` | 单元测试和行为契约验证。 |
| `ruff` | Python 代码 lint 和格式相关检查。 |
| `mypy` | Python 静态类型检查。 |
| `pre-commit` | Git 提交前自动检查框架。 |
| `nvitop` | 交互式 GPU 使用情况监控。 |
| `ninja` | 快速构建工具，常用于扩展编译。 |
| `git-lfs` | Git 大文件版本管理。 |

系统中已检测到可用：

| 工具 | 功能 |
|---|---|
| `git` | 版本控制。 |
| `cmake` | C/C++ 项目和扩展构建配置。 |
| `ffmpeg` | 视频、音频和图像序列处理。 |
| `curl` / `wget` | 命令行下载工具。 |
| `unzip` / `zip` | 压缩包解压和打包。 |
| `tree` | 目录树查看。 |
| `htop` | 交互式 CPU/内存/进程监控。 |
| `tmux` | 终端会话保持，适合长任务。 |
| `rsync` | 文件同步和增量复制。 |

## 4. 常见学习场景的工具选择

| 学习场景 | 推荐工具 | 说明 |
|---|---|---|
| 两三行 Tensor 形状、dtype、device 验证 | `ipython` | 反馈最快，适合查 shape、stride、广播、索引。 |
| 一行环境检查 | `python -c` | 适合快速检查 CUDA、版本、导入是否正常。 |
| 多行临时 smoke test | shell here-document | 不创建文件即可跑多行验证。 |
| 终端连接 Jupyter kernel | `jupyter console --kernel torch` | 适合想用 Jupyter kernel 但不打开浏览器的场景。 |
| 单函数/API 学习 | 小 `.py` + IPython `%run` | 代码开始需要保存、复跑、比较时使用。 |
| `nn.Module`、Hook、Loss、梯度实验 | `.py` + `pytest` + `torchinfo` | 适合验证模块输入输出契约和梯度。 |
| 数据探索、可视化、错误样本分析 | JupyterLab / Notebook | 适合图像展示、样本分析、日志读取。 |
| 最小训练闭环 | 训练脚本 + TensorBoard | 适合掌握 DataLoader 到优化器 step 的完整链路。 |
| 完整可复现实验 | 标准项目结构 + YAML + checkpoint + Git | 适合真实数据集、验证、保存、恢复和复现实验。 |
| 长时间任务 | `tmux` + 日志 + `nvitop` | 适合防止终端断开并观察 GPU。 |
| 性能定位 | `torch.profiler` + `nvitop` | 适合定位数据加载、GPU 利用率和同步瓶颈。 |

常用启动命令：

```bash
conda activate torch
ipython
```

```bash
conda activate torch
jupyter lab
```

```bash
conda activate torch
jupyter console --kernel torch
```

```bash
conda activate torch
tensorboard --logdir ~/runs
```

```bash
conda activate torch
nvitop
```

## 5. 需要手动执行的 sudo 操作

自动执行时，`sudo` 因需要交互式密码而失败。以下命令需要你手动输入 Linux 用户密码验证。

### 5.1 安全提醒

不要在 WSL2 中安装 NVIDIA Linux 显示驱动，也不要安装系统级 CUDA 驱动包。

不要执行：

```bash
sudo apt install nvidia-driver-*
sudo apt install cuda-drivers
sudo apt install cuda
sudo apt install cuda-12-*
sudo apt install cuda-13-*
```

当前 PyTorch GPU 环境依赖的是 Windows 主机 NVIDIA 驱动映射给 WSL2 的 CUDA Driver，以及 `torch` 环境中的官方 PyTorch CUDA wheel。

### 5.2 手动更新 apt 索引

```bash
sudo apt update
```

不建议为了本环境默认执行：

```bash
sudo apt upgrade -y
```

### 5.3 手动安装系统基础工具

```bash
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  git \
  git-lfs \
  curl \
  wget \
  ca-certificates \
  unzip \
  zip \
  tree \
  htop \
  tmux \
  rsync \
  ffmpeg \
  graphviz \
  libgl1 \
  libglib2.0-0
```

说明：

- `build-essential`：提供 gcc/g++/make 等编译基础工具。
- `cmake`、`ninja-build`、`pkg-config`：用于构建 C/C++ 或 Python 扩展。
- `git`、`git-lfs`：版本控制和大文件支持。
- `curl`、`wget`、`ca-certificates`：网络下载和证书支持。
- `unzip`、`zip`：数据集压缩包处理。
- `tree`、`htop`、`tmux`、`rsync`：日常开发、监控、长任务和同步工具。
- `ffmpeg`：视频/图像序列处理。
- `graphviz`：计算图渲染工具。
- `libgl1`、`libglib2.0-0`：OpenCV、图像库常见运行依赖。

### 5.4 手动初始化 Git LFS

系统级 `git-lfs` 安装后，可以执行：

```bash
git lfs install
```

如果你只在 `conda activate torch` 后使用 Git LFS，目前 `torch` 环境内已经有 `git-lfs`，并已执行过初始化。

### 5.5 手动验证系统工具

```bash
git --version
git lfs version
cmake --version
ninja --version
ffmpeg -version | head -n 1
dot -V
```

也可以一次性检查命令路径：

```bash
for cmd in git git-lfs cmake ninja ffmpeg dot curl wget unzip zip tree htop tmux rsync; do
  printf "%-8s " "$cmd"
  command -v "$cmd" || true
done
```

如果没有手动安装系统版 `ninja`、`dot` 或 `git-lfs`，但已激活 `torch` 环境，则它们应来自：

```text
/home/cyrus/anaconda3/envs/torch/bin/ninja
/home/cyrus/anaconda3/envs/torch/bin/dot
/home/cyrus/anaconda3/envs/torch/bin/git-lfs
```

### 5.6 手动验证 GPU 与 PyTorch

```bash
nvidia-smi
```

```bash
conda activate torch
python ~/workspace/pytorch-playground/verify_torch_env.py
python ~/workspace/pytorch-playground/verify_fashion_mnist.py
python -m pip check
```

预期关键输出：

```text
GPU name          : NVIDIA GeForce RTX 5080
PyTorch           : 2.12.1+cu130
torchvision       : 0.27.1+cu130
PyTorch CUDA      : 13.0
ALL CORE CHECKS PASSED
FASHION-MNIST SMOKE TEST PASSED
No broken requirements found.
```

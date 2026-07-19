# Codex 执行手册：在 WSL2 中自动搭建 PyTorch `torch` Conda 环境

> 本文面向 Codex，而不是仅供人工阅读。  
> Codex 应按阶段执行、记录输出、遇错停止，不得跳过关键验证。  
> 目标硬件：NVIDIA GeForce RTX 5080  
> 目标系统：Windows + WSL2 Ubuntu  
> 目标环境：Conda 环境 `torch`  
> 推荐核心版本：Python 3.11、PyTorch 2.12.1、torchvision 0.27.1、官方 CUDA 13.0 Wheel

---

# 0. 任务目标与边界

建立一个长期使用的通用 PyTorch 学习环境，支持：

- IPython 即时代码验证；
- JupyterLab；
- Tensor、Autograd 与 `nn.Module` 学习；
- Dataset/DataLoader；
- 真实图像数据训练；
- TensorBoard；
- 基础计算机视觉；
- 单元测试、静态检查；
- GPU 监控；
- 环境导出与复现。

本环境不用于强依赖特定 CUDA/C++ 扩展的科研项目。以下组件默认不安装：

```text
flash-attn
xformers
bitsandbytes
deepspeed
mmcv
mmdetection
mmsegmentation
MinkowskiEngine
spconv
pytorch3d
kaolin
torch-scatter
torch-sparse
3D Gaussian Splatting 自定义扩展
```

这些组件应在项目专用 Conda 环境中安装。

---

# 1. 不可违反的安全约束

## 1.1 不得在 WSL2 中安装 NVIDIA Linux 显示驱动

禁止执行：

```bash
sudo apt install nvidia-driver-*
sudo apt install cuda-drivers
sudo apt install cuda
sudo apt install cuda-12-*
sudo apt install cuda-13-*
```

WSL2 使用 Windows 主机 NVIDIA 驱动映射出的 CUDA Driver。普通 PyTorch Wheel 不要求安装系统 CUDA Toolkit。

如果未来需要 `nvcc` 编译自定义 CUDA 扩展，只能在用户明确授权后安装**不带驱动**的 CUDA Toolkit 包。本次任务不安装 Toolkit。

## 1.2 不得破坏现有 Conda 环境

禁止：

```bash
conda env remove -n torch
rm -rf ~/miniconda3
rm -rf ~/anaconda3
rm -rf ~/miniforge3
conda clean --all -y
```

若 `torch` 已存在：

- 健康且版本匹配：复用；
- 核心版本不匹配：停止并报告，建议新建 `torch-new`；
- 环境损坏：停止并报告；
- 不得静默删除重建。

## 1.3 不得混装 PyTorch

PyTorch 核心包统一使用官方 `pip` CUDA Wheel。

禁止在同一环境中无计划执行：

```bash
conda install pytorch
conda install pytorch-cuda
pip install torch
```

必须使用本文指定的固定版本和官方 CUDA 索引。

## 1.4 不得未经授权修改系统级配置

未经用户明确授权，不修改：

```text
/etc/wsl.conf
Windows 用户目录中的 .wslconfig
系统代理或 DNS
Windows NVIDIA 驱动
Shell 默认类型
Conda 根安装目录
```

## 1.5 所有安装必须可追踪

```bash
mkdir -p ~/setup-logs/torch-env
```

关键日志：

```text
~/setup-logs/torch-env/preflight.txt
~/setup-logs/torch-env/install.txt
~/setup-logs/torch-env/verify.txt
~/setup-logs/torch-env/freeze.txt
```

---

# 2. 目标状态

```text
Conda environment : torch
Python            : 3.11.x
PyTorch           : 2.12.1
torchvision       : 0.27.1
PyTorch CUDA      : 13.0 build
CUDA available    : True
GPU               : NVIDIA GeForce RTX 5080
Jupyter kernel    : Python (torch)
Project root      : ~/workspace
Dataset root      : ~/datasets
Checkpoint root   : ~/checkpoints
Run log root      : ~/runs
```

注意：

- `nvidia-smi` 显示的 CUDA Version 是驱动支持上限；
- `torch.version.cuda` 表示 PyTorch Wheel 使用的 CUDA Runtime；
- `nvcc` 不存在不代表 PyTorch GPU 环境异常。

---

# 3. Codex 执行协议

按以下阶段执行：

```text
Phase 0：只读检查
Phase 1：系统基础包
Phase 2：Conda 环境
Phase 3：PyTorch 核心包
Phase 4：通用 Python 工具
Phase 5：Jupyter 与目录
Phase 6：离线核心验证
Phase 7：联网数据集验证
Phase 8：环境冻结与报告
```

每阶段要求：

1. 执行前说明目标；
2. 命令失败立即停止当前阶段；
3. 不用后续安装掩盖前面错误；
4. 保存关键输出；
5. 阶段完成后报告 PASS/FAIL；
6. 不得仅凭安装命令退出码判断成功，必须实际导入和计算。

---

# 4. Phase 0：只读前置检查

## 4.1 收集系统信息

```bash
mkdir -p ~/setup-logs/torch-env

{
  echo "=== DATE ==="
  date --iso-8601=seconds

  echo
  echo "=== USER / SHELL ==="
  whoami
  echo "$SHELL"
  pwd

  echo
  echo "=== OS ==="
  uname -a
  cat /etc/os-release

  echo
  echo "=== WSL ==="
  grep -i microsoft /proc/version || true
  wslpath -w "$HOME" || true

  echo
  echo "=== GPU ==="
  command -v nvidia-smi || true
  nvidia-smi || true

  echo
  echo "=== CONDA ==="
  command -v conda || true
  conda --version || true
  conda info || true
  conda env list || true

  echo
  echo "=== STORAGE ==="
  df -h "$HOME"
  df -h /mnt/c 2>/dev/null || true

  echo
  echo "=== MEMORY ==="
  free -h

  echo
  echo "=== CPU ==="
  nproc
  lscpu | sed -n '1,25p'
} 2>&1 | tee ~/setup-logs/torch-env/preflight.txt
```

## 4.2 判定规则

必须识别：

- WSL2/WSL 环境；
- `nvidia-smi` 可执行；
- GPU 名称包含 RTX 5080；
- Conda 可用；
- Home 目录空间足够；
- `torch` 环境是否已存在。

若 `nvidia-smi` 失败：

1. 不安装 PyTorch；
2. 不在 WSL 中安装驱动；
3. 报告 Windows 驱动/WSL 状态；
4. 建议用户在 Windows PowerShell 执行：

```powershell
wsl --update
wsl --shutdown
nvidia-smi
```

若 Conda 不可用，检查常见路径：

```bash
ls -d ~/miniconda3 ~/anaconda3 ~/miniforge3 2>/dev/null || true
```

尝试加载：

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null
```

仍不可用则停止，不擅自安装第二套 Conda。

---

# 5. Phase 1：系统基础工具

## 5.1 更新索引

```bash
sudo apt update
```

不默认执行 `sudo apt upgrade -y`，因为它可能引入较大系统变更。

## 5.2 安装工具

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

```bash
git lfs install
```

验证：

```bash
{
  git --version
  git lfs version
  cmake --version
  ninja --version
  ffmpeg -version | head -n 1
  dot -V
} 2>&1 | tee -a ~/setup-logs/torch-env/install.txt
```

---

# 6. Phase 2：创建或检查 Conda 环境

## 6.1 加载 Conda

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
```

## 6.2 检查环境

```bash
if conda env list | awk '{print $1}' | grep -qx "torch"; then
  echo "Environment 'torch' already exists."
else
  conda create -n torch python=3.11 pip -y
fi
```

## 6.3 激活并检查

```bash
conda activate torch

which python
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
```

目标为 Python 3.11。若现有 `torch` 不是 3.11：

- 不直接替换 Python；
- 不删除环境；
- 停止并报告；
- 推荐用户授权后新建 `torch-new`。

## 6.4 升级安装基础设施

```bash
python -m pip install --upgrade pip setuptools wheel
```

确认解释器属于目标环境：

```bash
python - <<'PY'
import sys
import pip

print("python:", sys.executable)
print("pip:", pip.__file__)
assert "/envs/torch/" in sys.executable, sys.executable
PY
```

---

# 7. Phase 3：安装 PyTorch 核心包

## 7.1 固定版本

```text
torch       2.12.1
torchvision 0.27.1
CUDA Wheel  cu130
```

官方安装命令：

```bash
python -m pip install \
  torch==2.12.1 \
  torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

默认不安装 `torchaudio`。需要音频任务时再单独处理。

## 7.2 版本检查

```bash
python - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA runtime:", torch.version.cuda)

assert torch.__version__.startswith("2.12.1")
assert torchvision.__version__.startswith("0.27.1")
assert torch.version.cuda is not None
PY
```

若安装成 CPU 版或版本错误，可只卸载核心包：

```bash
python -m pip uninstall -y torch torchvision
```

然后重新执行官方 `cu130` 安装命令。

---

# 8. Phase 4：安装通用学习工具

## 8.1 科学计算与交互

```bash
python -m pip install \
  ipython \
  numpy \
  scipy \
  pandas \
  scikit-learn \
  matplotlib \
  pillow \
  tqdm \
  rich \
  pyyaml \
  psutil
```

## 8.2 Jupyter

```bash
python -m pip install \
  jupyterlab \
  ipykernel \
  ipywidgets
```

## 8.3 PyTorch 工具

```bash
python -m pip install \
  tensorboard \
  torchinfo \
  torchmetrics \
  einops \
  torchviz
```

## 8.4 计算机视觉

```bash
python -m pip install \
  timm \
  opencv-python-headless \
  albumentations \
  scikit-image
```

不得同时安装：

```text
opencv-python
opencv-python-headless
```

若检测到两者共存，只报告，不自动卸载用户包。

## 8.5 数据格式

```bash
python -m pip install \
  h5py \
  pyarrow
```

## 8.6 测试与代码质量

```bash
python -m pip install \
  pytest \
  ruff \
  mypy \
  pre-commit
```

## 8.7 GPU 监控

```bash
python -m pip install nvitop
```

## 8.8 依赖检查

```bash
python -m pip check | tee -a ~/setup-logs/torch-env/install.txt
```

目标：

```text
No broken requirements found.
```

若有冲突，停止后续阶段并报告依赖链。

---

# 9. 可选安装组

不得默认安装，仅在用户明确要求时执行。

## 9.1 Hugging Face

```bash
python -m pip install \
  transformers \
  datasets \
  accelerate \
  safetensors \
  sentencepiece \
  huggingface-hub
```

## 9.2 实验管理

```bash
python -m pip install \
  wandb \
  optuna
```

不强制执行 WandB 登录。

## 9.3 ONNX

```bash
python -m pip install \
  onnx \
  onnxruntime
```

默认不安装 `onnxruntime-gpu`，避免额外 CUDA Runtime 耦合。

---

# 10. Phase 5：标准目录与 Jupyter Kernel

## 10.1 创建目录

```bash
mkdir -p \
  ~/workspace \
  ~/datasets \
  ~/checkpoints \
  ~/runs \
  ~/workspace/pytorch-playground
```

检查不位于 Windows 挂载盘：

```bash
for path in \
  "$HOME/workspace" \
  "$HOME/datasets" \
  "$HOME/checkpoints" \
  "$HOME/runs"
do
  case "$path" in
    /mnt/*)
      echo "ERROR: $path is on a mounted Windows drive."
      exit 1
      ;;
    *)
      echo "PASS: $path"
      ;;
  esac
done
```

## 10.2 注册 Kernel

```bash
python -m ipykernel install \
  --user \
  --name torch \
  --display-name "Python (torch)"
```

```bash
jupyter kernelspec list
```

---

# 11. Phase 6：离线核心验证

创建：

```text
~/workspace/pytorch-playground/verify_torch_env.py
```

内容：

```python
from __future__ import annotations

import platform
import sys

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import nms


def section(name: str) -> None:
    print(f"\n{'=' * 20} {name} {'=' * 20}")


def verify_versions() -> None:
    section("VERSIONS")
    print("Python executable :", sys.executable)
    print("Python version    :", sys.version.split()[0])
    print("Platform          :", platform.platform())
    print("NumPy             :", np.__version__)
    print("PyTorch           :", torch.__version__)
    print("torchvision       :", torchvision.__version__)
    print("PyTorch CUDA      :", torch.version.cuda)
    print("cuDNN             :", torch.backends.cudnn.version())

    assert torch.__version__.startswith("2.12.1")
    assert torchvision.__version__.startswith("0.27.1")


def verify_cuda() -> torch.device:
    section("CUDA")
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)

    print("GPU count         :", torch.cuda.device_count())
    print("GPU name          :", name)
    print("Compute capability:", capability)
    print("Supported archs   :", torch.cuda.get_arch_list())
    print("BF16 supported    :", torch.cuda.is_bf16_supported())

    assert "5080" in name, f"Unexpected GPU: {name}"
    return device


def verify_autograd(device: torch.device) -> None:
    section("AUTOGRAD")
    x = torch.randn(1024, 1024, device=device, requires_grad=True)
    y = x @ x.T
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    print("loss         :", loss.item())
    print("grad finite  :", torch.isfinite(x.grad).all().item())
    print("allocated MB :", torch.cuda.memory_allocated() / 1024**2)


def verify_module(device: torch.device) -> None:
    section("MODULE")
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.GELU(),
        nn.Linear(256, 10),
    ).to(device)

    inputs = torch.randn(32, 128, device=device)
    labels = torch.randint(0, 10, (32,), device=device)

    logits = model(inputs)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert logits.shape == (32, 10)
    assert model[0].weight.grad is not None
    print("logits shape :", tuple(logits.shape))
    print("loss         :", loss.item())


def verify_torchvision_ops(device: torch.device) -> None:
    section("TORCHVISION CUDA OPS")
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        device=device,
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    keep = nms(boxes, scores, iou_threshold=0.5)

    assert keep.device.type == "cuda"
    print("NMS result:", keep.tolist())


def verify_bfloat16(device: torch.device) -> None:
    section("BF16")
    if not torch.cuda.is_bf16_supported():
        print("SKIP: BF16 is not reported as supported.")
        return

    model = nn.Linear(128, 64).to(device)
    x = torch.randn(16, 128, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x)
        loss = y.square().mean()

    loss.backward()
    print("output dtype:", y.dtype)
    assert y.dtype == torch.bfloat16


def verify_dataloader(device: torch.device) -> None:
    section("DATALOADER")
    features = torch.randn(256, 32)
    labels = torch.randint(0, 4, (256,))
    dataset = TensorDataset(features, labels)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    batch_x, batch_y = next(iter(loader))
    batch_x = batch_x.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)

    assert batch_x.shape == (32, 32)
    assert batch_y.shape == (32,)
    print("batch device:", batch_x.device)


def main() -> None:
    verify_versions()
    device = verify_cuda()
    verify_autograd(device)
    verify_module(device)
    verify_torchvision_ops(device)
    verify_bfloat16(device)
    verify_dataloader(device)

    section("RESULT")
    print("ALL CORE CHECKS PASSED")


if __name__ == "__main__":
    main()
```

运行：

```bash
conda activate torch
python ~/workspace/pytorch-playground/verify_torch_env.py \
  2>&1 | tee ~/setup-logs/torch-env/verify.txt
```

成功标志：

```text
ALL CORE CHECKS PASSED
```

必须同时验证 CUDA 运算、反向传播、torchvision CUDA 算子、BF16 和 DataLoader。

---

# 12. Phase 7：联网数据集验证

创建：

```text
~/workspace/pytorch-playground/verify_fashion_mnist.py
```

内容：

```python
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    dataset = datasets.FashionMNIST(
        root=Path.home() / "datasets",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    images, labels = next(iter(loader))
    print("dataset size:", len(dataset))
    print("images:", images.shape, images.dtype)
    print("labels:", labels.shape, labels.dtype)
    print("range:", images.min().item(), images.max().item())

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    optimizer.step()

    assert logits.shape == (64, 10)
    assert torch.isfinite(loss)
    print("loss:", loss.item())
    print("FASHION-MNIST SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
```

运行：

```bash
python ~/workspace/pytorch-playground/verify_fashion_mnist.py \
  2>&1 | tee -a ~/setup-logs/torch-env/verify.txt
```

若下载失败，应区分网络问题与 PyTorch 运行问题。

---

# 13. Phase 8：冻结环境

## 13.1 精确 pip 快照

```bash
python -m pip freeze \
  > ~/setup-logs/torch-env/requirements-lock.txt
```

## 13.2 Conda 完整导出

```bash
conda env export --no-builds \
  > ~/setup-logs/torch-env/environment.yml
```

## 13.3 Conda 最小历史导出

```bash
conda env export --from-history \
  > ~/setup-logs/torch-env/environment-minimal.yml
```

## 13.4 环境摘要

```bash
{
  echo "=== CONDA ==="
  conda env list

  echo
  echo "=== PYTHON ==="
  which python
  python --version

  echo
  echo "=== PIP CHECK ==="
  python -m pip check

  echo
  echo "=== CORE PACKAGES ==="
  python -m pip show torch torchvision numpy ipython jupyterlab tensorboard

  echo
  echo "=== GPU ==="
  python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
} 2>&1 | tee ~/setup-logs/torch-env/freeze.txt
```

---

# 14. 最终验收清单

Codex 应逐项报告：

```text
[ ] WSL 环境确认
[ ] nvidia-smi 可用
[ ] RTX 5080 被识别
[ ] 未安装 Linux NVIDIA 驱动
[ ] Conda 环境 torch 激活
[ ] Python 3.11
[ ] torch 2.12.1
[ ] torchvision 0.27.1
[ ] torch.version.cuda 为 13.0
[ ] torch.cuda.is_available() 为 True
[ ] CUDA 矩阵计算通过
[ ] Autograd 通过
[ ] torchvision NMS CUDA 算子通过
[ ] BF16 通过或明确 SKIP 原因
[ ] DataLoader 通过
[ ] Fashion-MNIST Smoke Test 通过
[ ] Jupyter Kernel 已注册
[ ] pip check 无冲突
[ ] requirements-lock.txt 已生成
[ ] environment.yml 已生成
[ ] environment-minimal.yml 已生成
```

---

# 15. 故障分流

## 15.1 `nvidia-smi` 在 WSL 不可用

禁止安装 Linux 驱动。建议在 Windows PowerShell 检查：

```powershell
nvidia-smi
wsl --update
wsl --shutdown
```

## 15.2 `torch.cuda.is_available()` 为 False

```bash
python -m pip show torch
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

常见原因：CPU Wheel、解释器错误、Windows 驱动过旧、WSL 未重启、PyTorch 混装。

## 15.3 `operator torchvision::nms does not exist`

检查：

```bash
python -m pip show torch torchvision
```

目标：

```text
torch       2.12.1
torchvision 0.27.1
```

## 15.4 `no kernel image is available`

检查：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_arch_list())
print(torch.cuda.get_device_capability())
PY
```

RTX 5080 应使用支持其架构的现代 CUDA Wheel；本手册固定 `cu130`。

## 15.5 DataLoader Worker 异常

改为：

```python
num_workers=0
persistent_workers=False
```

单进程正常后再增加到 2、4。

## 15.6 下载失败

```bash
curl -I https://download.pytorch.org
python -c "import ssl; print(ssl.OPENSSL_VERSION)"
```

不得全局关闭 SSL 验证。

## 15.7 pip 冲突

```bash
python -m pip check
python -m pip list
```

报告冲突链，不使用大规模 Eager Upgrade 作为默认修复。

---

# 16. 推荐项目级 `AGENTS.md`

```markdown
# PyTorch Playground Instructions

## Environment

- Work inside WSL2.
- Use the Conda environment named `torch`.
- Before executing Python, run:
  `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate torch`
- Use `python -m pip`, not bare `pip`.
- Store projects under `~/workspace` and datasets under `~/datasets`.
- Do not store training datasets under `/mnt/c` unless explicitly requested.

## CUDA safety

- Never install an NVIDIA Linux display driver inside WSL2.
- Never run `apt install nvidia-driver-*`, `cuda-drivers`, or the `cuda` meta-package.
- A missing `nvcc` is not an error for ordinary PyTorch use.
- Before changing PyTorch versions, report Python, torch, torchvision, and CUDA runtime versions.

## Code execution

- For temporary checks, prefer a short reproducible Python snippet.
- For code longer than roughly 20 lines, create a `.py` file.
- Add assertions for shape, dtype, device, finite values, and gradients.
- Before full training, run:
  1. import test;
  2. one-sample Dataset test;
  3. one-Batch DataLoader test;
  4. one forward/backward step;
  5. one-Batch overfit test.

## Validation

- After dependency changes, run `python -m pip check`.
- After PyTorch changes, run `verify_torch_env.py`.
- Run `pytest -q` when tests exist.
- Do not claim success from installation output alone.
```

---

# 17. 可直接交给 Codex 的提示词

```text
请阅读当前目录中的《Codex 执行手册：在 WSL2 中自动搭建 PyTorch torch Conda 环境》并按阶段执行。

目标是在当前 WSL2 Ubuntu 中建立或验证名为 torch 的 Conda 环境，用于 RTX 5080 上的 PyTorch 学习和中小型数据训练。

严格要求：
1. 先执行 Phase 0 只读检查并汇报，不要立即安装。
2. 不得在 WSL 中安装 NVIDIA Linux 驱动、cuda-drivers、cuda 或 cuda-xx 元包。
3. 不得删除已有 Conda 环境。
4. 若 torch 已存在，先检查 Python、torch、torchvision 和 CUDA Runtime；重大不一致时停止说明，不静默重建。
5. 分阶段执行并将日志保存到 ~/setup-logs/torch-env。
6. PyTorch 固定使用官方 CUDA 13.0 Wheel：torch==2.12.1，torchvision==0.27.1。
7. 每阶段给出 PASS/FAIL。
8. 必须运行 GPU、Autograd、torchvision NMS、BF16、DataLoader 离线验证。
9. 网络正常时运行 Fashion-MNIST Smoke Test。
10. 最后运行 pip check，并导出 requirements-lock.txt、environment.yml、environment-minimal.yml。
11. 对失败给出根因和最小修复，不以删除环境或批量升级作为默认修复。
```

---

# 18. 参考资料

1. PyTorch 官方安装页：  
   https://pytorch.org/get-started/locally/

2. PyTorch 官方历史版本与 CUDA Wheel：  
   https://pytorch.org/get-started/previous-versions/

3. NVIDIA CUDA on WSL User Guide：  
   https://docs.nvidia.com/cuda/wsl-user-guide/index.html

4. Microsoft WSL 文件系统性能建议：  
   https://learn.microsoft.com/en-us/windows/wsl/filesystems

5. OpenAI Codex `AGENTS.md` 官方说明：  
   https://developers.openai.com/codex/guides/agents-md

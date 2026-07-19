# PyTorch 与 AI 初学者：基于代码实操的分层学习指南

> 适用环境：Windows + WSL2 + Conda + NVIDIA GPU  
> 适用对象：具备基础 Python/Linux 知识，希望通过代码验证、模块实验和真实训练系统学习 PyTorch 的初学者  
> 核心原则：**根据任务粒度、反馈周期和复现要求选择工具，不要把所有学习活动都塞进 Notebook、单个脚本或远程服务器。**

---

## 1. 学习任务的六级模型

| 层级 | 典型耗时 | 典型任务 | 推荐载体 | GPU |
|---|---:|---|---|---|
| L0 即时验证 | 10 秒～3 分钟 | Tensor 形状、索引、广播、函数返回值 | IPython、`python -c` | 通常不需要 |
| L1 函数实验 | 5～30 分钟 | 单个函数、Loss、Optimizer 行为 | 小型 `.py` + IPython `%run` | 按需 |
| L2 模块实验 | 20～90 分钟 | `nn.Module`、Attention、卷积块、Hook | 模块化脚本 + `pytest` | 按需 |
| L3 最小训练闭环 | 30 分钟～3 小时 | 单 Batch 过拟合、小数据集训练 | 训练脚本 + TensorBoard | 推荐 |
| L4 完整可复现实验 | 数小时～数天 | 真实数据集、验证、保存、恢复、指标 | 标准项目结构 + 配置文件 | 推荐 |
| L5 性能与科研实验 | 数小时～数周 | Profiling、混合精度、多卡、消融 | 项目环境 + 服务器 | 通常需要 |

不应使用同一种工具完成所有层级：

- 为验证 `torch.stack` 写完整训练项目，成本过高；
- 在 IPython 中手工完成数小时训练，难以复现；
- 为查看一个 Tensor 的 `stride` 登录远程服务器，反馈过慢；
- 把完整科研代码放进一个 Notebook，会产生隐式状态、结构混乱和测试困难。

---

# 2. 场景一：临时、即兴的几行代码验证

## 2.1 适用任务

- `reshape`、`view`、`permute` 的差异；
- `torch.cat` 与 `torch.stack` 的输出形状；
- 广播规则；
- 布尔索引与高级索引；
- `dtype`、`device`、`requires_grad`；
- Loss 对输入形状和标签类型的要求；
- `softmax` 应在哪个维度计算；
- 某个操作是否可求梯度；
- CPU/GPU 返回值和精度差异；
- 连续内存、`stride` 和 `contiguous`。

## 2.2 首选工具：IPython

日常即时验证推荐默认进入 IPython：

```bash
conda activate torch
ipython
```

IPython 相比普通 Python REPL 的主要优势：

- Tab 自动补全；
- `?` 和 `??` 查看接口、文档与源码；
- `%timeit` 快速计时；
- `%run` 执行脚本；
- `%load` 加载代码；
- 更清晰的异常堆栈；
- 自动保存命令历史。

示例：

```python
import torch

x = torch.arange(24).reshape(2, 3, 4)
x.shape
x.stride()
x.is_contiguous()

torch.stack([x, x], dim=0).shape
torch.cat([x, x], dim=0).shape
```

查看 API：

```python
torch.permute?
torch.nn.CrossEntropyLoss??
```

快速计时：

```python
x = torch.randn(4096, 4096, device="cuda")
%timeit x @ x
```

运行当前目录脚本：

```python
%run tensor_lab.py
```

## 2.3 何时应离开 IPython

出现以下任一情况，应保存为 `.py`：

1. 代码超过约 15～30 行；
2. 需要重复运行或比较多个设置；
3. 需要断言、测试、随机种子；
4. 出现函数、类或训练循环；
5. 需要让 Codex 修改；
6. 需要提交 Git；
7. 需要定位具体行号；
8. 结果以后还会查阅。

IPython 是**实验台**，不是长期代码仓库。

## 2.4 其他快速方式

### 单行命令

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

适合环境和版本检查。

### Shell Here-document

```bash
python - <<'PY'
import torch

x = torch.randn(2, 3, device="cuda")
print(x)
print(x.shape, x.dtype, x.device)
PY
```

适合无需创建文件的多行 Smoke Test（冒烟测试）。

### VS Code 局部执行

在 WSL 窗口中选中代码，发送到 Python Terminal 或 Interactive Window。适合阅读脚本时局部实验，但最终结果仍应回到可从头运行的文件。

---

# 3. 即时验证的标准模板

不要只打印 Tensor 数值，至少检查形状、类型、设备、内存布局和梯度属性：

```python
import torch

x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
y = x.permute(0, 2, 1)

print("x.shape        =", x.shape)
print("y.shape        =", y.shape)
print("x.dtype        =", x.dtype)
print("x.device       =", x.device)
print("x.stride       =", x.stride())
print("y.stride       =", y.stride())
print("x.contiguous   =", x.is_contiguous())
print("y.contiguous   =", y.is_contiguous())
print("requires_grad  =", x.requires_grad)

assert y.shape == (2, 4, 3)
```

固定检查清单：

```text
值是否符合预期？
形状是否符合预期？
数据类型是否正确？
设备是否正确？
内存布局是否变化？
是否保留梯度链？
边界输入是否正常？
```

---

# 4. 场景二：函数、算子和 API 学习

## 4.1 推荐 Playground 结构

```text
~/workspace/pytorch-playground/
├── 01_tensor/
├── 02_autograd/
├── 03_nn_functions/
├── 04_nn_modules/
├── 05_dataloader/
├── 06_training/
├── 07_vision/
├── 08_transformer/
├── 09_quantization/
├── notebooks/
└── tests/
```

每个知识点一个短文件：

```text
01_tensor/
├── reshape_vs_view.py
├── cat_vs_stack.py
├── broadcasting.py
├── advanced_indexing.py
└── contiguous_and_stride.py
```

## 4.2 函数学习八步法

### 第 1 步：理解签名

回答：输入、返回值、默认参数、维度规则、是否原地、是否支持 Batch。

```python
torch.flatten?
torch.flatten??
```

### 第 2 步：构造最小且可手算输入

```python
x = torch.arange(12).reshape(3, 4)
```

可手算输入通常比随机输入更利于理解。

### 第 3 步：验证 Shape Contract（形状契约）

```python
y = torch.flatten(x, start_dim=0)
assert y.shape == (12,)
```

### 第 4 步：验证 `dtype` 与 `device`

```python
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    x = torch.randn(2, 3, device="cuda", dtype=dtype)
    y = torch.softmax(x, dim=-1)
    print(dtype, y.dtype, y.device)
```

### 第 5 步：验证边界和错误输入

至少考虑：

- Batch Size 为 1；
- 空维度；
- 极大/极小数；
- 非连续 Tensor；
- 不合法维度；
- 整数输入；
- CPU/GPU；
- 混合精度。

### 第 6 步：验证梯度

```python
x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)

def fn(inp):
    return torch.sin(inp) * inp

assert torch.autograd.gradcheck(fn, (x,))
```

`gradcheck` 使用数值差分检查解析梯度，常用双精度输入。

### 第 7 步：验证性能

GPU 计时需要同步：

```python
import time
import torch

x = torch.randn(4096, 4096, device="cuda")

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    y = torch.relu(x)
torch.cuda.synchronize()

print("elapsed:", time.perf_counter() - start)
```

### 第 8 步：记录结论

```python
"""
结论：
1. permute 只改变维度顺序，不直接复制底层数据。
2. permute 后通常不是 contiguous。
3. view 要求兼容的内存布局，reshape 必要时会复制。
"""
```

留下机制结论，而不是只留下能运行的代码。

---

# 5. 场景三：`nn.Module` 和网络模块学习

## 5.1 典型对象

- `nn.Linear`、`nn.Conv2d`；
- BatchNorm、LayerNorm、Dropout；
- Embedding、Multi-Head Attention（多头注意力）；
- 残差块、MLP Block、Transformer Block；
- 自定义 Loss；
- Observer（观测器）与 Fake Quantization（伪量化）模块。

## 5.2 工具组合

| 工具 | 用途 |
|---|---|
| `.py` 模块文件 | 保存类和实验入口 |
| `torchinfo` | 查看层输出与参数量 |
| Forward Hook | 检查中间特征 |
| `pytest` | 固化输入输出契约 |
| `torchviz` | 可视化小型计算图 |
| VS Code Debugger / `breakpoint()` | 逐步调试 Forward |
| `torch.autograd.detect_anomaly()` | 定位异常梯度 |
| `torch.profiler` | 分析算子时间和显存 |

## 5.3 模块实验模板

```python
from __future__ import annotations

import torch
from torch import nn
from torchinfo import summary


class MLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPBlock(dim=128).to(device)
    x = torch.randn(2, 16, 128, device=device)
    y = model(x)

    assert y.shape == x.shape
    assert y.device == x.device

    loss = y.square().mean()
    loss.backward()

    first_weight = model.net[0].weight
    assert first_weight.grad is not None
    assert torch.isfinite(first_weight.grad).all()

    summary(model, input_size=(2, 16, 128), device=str(device))
    print("output shape:", y.shape)
    print("loss:", loss.item())


if __name__ == "__main__":
    main()
```

## 5.4 Hook 检查中间特征

```python
def print_shape_hook(name: str):
    def hook(module, inputs, output):
        print(f"{name}: {tuple(inputs[0].shape)} -> {tuple(output.shape)}")
    return hook


handle = model.net[0].register_forward_hook(
    print_shape_hook("first_linear")
)
_ = model(x)
handle.remove()
```

Hook 适合观察，不应长期承载主要业务逻辑。

## 5.5 使用 `pytest`

```python
import torch
from mlp_block import MLPBlock


def test_preserves_shape() -> None:
    model = MLPBlock(dim=64)
    x = torch.randn(2, 8, 64)
    assert model(x).shape == x.shape


def test_backward() -> None:
    model = MLPBlock(dim=64)
    x = torch.randn(2, 8, 64, requires_grad=True)
    model(x).mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
```

运行：

```bash
pytest -q
```

---

# 6. 场景四：Dataset、DataLoader 与真实数据

## 6.1 本地 WSL2 适用范围

本地适合：

- MNIST、Fashion-MNIST、CIFAR-10；
- 小型猫狗分类；
- 少量语义分割图像；
- 自定义 `Dataset`；
- 数据增强可视化；
- DataLoader 多进程；
- 单 Batch 与小样本调试。

仅当数据过大、训练过久、显存不足或需要多 GPU 时迁移服务器。

## 6.2 数据目录

```text
~/datasets/
├── fashion-mnist/
├── cifar10/
├── cats-vs-dogs/
└── temporary/
```

在 WSL Linux 命令行训练时，高频读取的项目和数据优先放在：

```text
/home/<user>/workspace
/home/<user>/datasets
```

而不是 `/mnt/c`、`/mnt/d`。

## 6.3 DataLoader 逐级调试

### 1. 只访问 Dataset

```python
sample, label = train_dataset[0]
print(type(sample), sample.shape, label)
```

### 2. 先用单进程

```python
loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)
```

### 3. 检查一个 Batch

```python
images, labels = next(iter(loader))
print(images.shape, labels.shape)
print(images.dtype, labels.dtype)
print(images.min(), images.max())
```

### 4. 再启用多进程

```python
loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)
```

本地一般从 `2` 或 `4` 开始测试，不要盲目设为全部 CPU 核心。

### 5. 可视化反归一化结果

图像任务必须确认：

- 通道顺序；
- 标签与图片对应；
- 增强不破坏语义；
- 归一化参数；
- 分割掩码未使用错误插值。

---

# 7. 场景五：最小训练闭环

重点掌握完整数据流：

```text
DataLoader
→ CPU Tensor
→ GPU Tensor
→ Model Forward
→ Loss
→ Zero Grad
→ Backward
→ Optimizer Step
→ Validation
```

## 7.1 标准调试阶梯

### 单 Batch 前向

```python
images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)

logits = model(images)
loss = criterion(logits, labels)
print(images.shape, labels.shape, logits.shape, loss.item())
```

### 单 Batch 反向

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

检查梯度：

```python
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
```

### 过拟合一个 Batch

固定同一个 Batch 训练数百步。若 Loss 无法显著下降，优先检查：

- 标签映射；
- Loss 与输出是否匹配；
- 参数是否加入优化器；
- 梯度是否被 `detach`；
- 学习率；
- 数据本身。

### 过拟合小数据集

选 100～1000 个样本，验证 Dataset、DataLoader、Train 和 Validation 全链路。

### 完整训练

前三步通过后，再投入完整数据和长时间训练。

---

# 8. 场景六：完整可复现训练项目

## 8.1 推荐结构

```text
project/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── requirements-lock.txt
├── configs/
│   ├── baseline.yaml
│   └── debug.yaml
├── src/
│   └── project_name/
│       ├── __init__.py
│       ├── data.py
│       ├── models.py
│       ├── losses.py
│       ├── metrics.py
│       ├── train.py
│       └── utils.py
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   └── smoke_test.sh
├── tests/
├── notebooks/
├── outputs/
└── checkpoints/
```

职责：

- `src/`：可复用核心代码；
- `configs/`：实验参数；
- `scripts/`：稳定入口；
- `tests/`：契约与回归测试；
- `notebooks/`：分析与可视化；
- `outputs/`：日志、指标和图；
- `checkpoints/`：模型权重。

## 8.2 Notebook 的正确定位

Notebook 适合：

- 数据探索；
- 可视化；
- 逐步讲解；
- 错误样本分析；
- 读取日志；
- 展示预测结果。

Notebook 不应是唯一训练入口，因为单元格可乱序、状态隐式、难参数化、难测试、Git Diff 可读性差。

推荐：

```text
Notebook 负责分析
.py 负责训练
配置文件负责参数
测试负责正确性
```

## 8.3 训练脚本至少包含

- 随机种子；
- 设备选择；
- 数据路径；
- Batch Size、学习率、Epoch；
- 日志与验证；
- 最佳/最后模型保存；
- 断点恢复；
- 环境和 Git Commit 信息。

---

# 9. 调试工具分工

## 9.1 `print` 与断言

```python
print({
    "shape": tuple(x.shape),
    "dtype": str(x.dtype),
    "device": str(x.device),
    "min": x.min().item(),
    "max": x.max().item(),
    "mean": x.mean().item(),
})
```

```python
assert logits.ndim == 2
assert logits.shape[0] == labels.shape[0]
assert labels.dtype == torch.long
```

## 9.2 `breakpoint()` 与 VS Code Debugger

适合复杂 Forward、局部变量、条件断点与异常捕获。DataLoader 多进程出错时先设 `num_workers=0`。

## 9.3 异常梯度

```python
with torch.autograd.detect_anomaly():
    loss.backward()
```

只在调试时使用，因为性能开销较大。

## 9.4 GPU 监控

```bash
nvidia-smi
watch -n 1 nvidia-smi
nvitop
```

关注 GPU 利用率、显存、功耗、温度、其他进程占用和数据加载等待。

---

# 10. 实验记录

## 10.1 TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_001")
writer.add_scalar("train/loss", loss.item(), global_step)
writer.add_scalar("val/accuracy", accuracy, epoch)
writer.close()
```

```bash
tensorboard --logdir runs
```

建议记录：Loss、验证指标、学习率、梯度范数、示例图片和预测。

## 10.2 正式实验目录

```text
outputs/2026-07-16_143000_baseline/
├── config.yaml
├── train.log
├── metrics.json
├── environment.txt
├── git_commit.txt
├── checkpoints/
└── figures/
```

禁止依赖“我记得当时用了哪个参数”。

---

# 11. 性能分析

顺序应为：

```text
先正确
→ 再可复现
→ 再定位瓶颈
→ 最后优化
```

`torch.profiler` 适合定位 CPU 数据加载、GPU 算子、CPU-GPU 同步和显存问题。

常见瓶颈：

- GPU 利用率长期过低；
- 数据加载时间高于 Forward；
- 大量 `.item()` 导致同步；
- 循环中频繁 `.cpu()`；
- Batch Size 太小；
- 未使用 `pin_memory`；
- 数据位于 `/mnt/c`；
- 重复预处理；
- 意外保留完整计算图。

---

# 12. 本地 WSL2 与服务器边界

## 本地优先

- Tensor/API 验证；
- 模块单元测试；
- 数据可视化；
- 小型数据集；
- 单 Batch 过拟合；
- 轻量 CNN/MLP/ViT；
- 调试、分析和笔记。

## 服务器优先

- 显存超过本地 GPU；
- 多 GPU；
- 大型 2D/3D 分割；
- 大规模预训练；
- 多随机种子、网格搜索；
- 数小时至数天的正式实验；
- 大规模数据集；
- 论文消融实验。

## 推荐迁移流程

```text
本地写代码
→ 静态检查
→ 单元测试
→ 单 Batch
→ 小数据过拟合
→ Git 提交
→ 服务器拉取
→ 服务器 Smoke Test
→ 正式训练
```

---

# 13. 工具矩阵

| 任务 | 首选 | 辅助 |
|---|---|---|
| 两三行 Tensor 验证 | IPython | `python -c` |
| 多行临时代码 | Here-document | 临时 `.py` |
| 单函数学习 | 小 `.py` | IPython `%run` |
| 模块学习 | `.py` + `pytest` | `torchinfo`、Hook |
| 数据探索 | Jupyter | Matplotlib |
| 训练主流程 | `.py` | YAML、TensorBoard |
| 错误定位 | VS Code Debugger | `breakpoint()` |
| 梯度问题 | 断言、Anomaly Detection | `gradcheck` |
| 性能问题 | `torch.profiler` | `nvitop` |
| 正式实验 | 标准项目结构 | Git、配置、日志 |
| 远程长任务 | `tmux` | `nvitop`、日志 |

---

# 14. 推荐学习路线

1. **Tensor 与 NumPy 对照**：Shape、Indexing、Broadcasting、Reduction、MatMul、Memory Layout、Device、Dtype。
2. **Autograd 与优化**：标量导数、计算图、梯度累积、`detach`、手写梯度下降、Optimizer。
3. **模块**：MLP、CNN、Normalization、Dropout、Residual、Attention、Transformer Block。
4. **数据与训练闭环**：Dataset、DataLoader、Transform、Train、Validation、Checkpoint、Resume。
5. **迁移学习**：torchvision/timm、冻结与解冻、参数组、调度器、混合精度、误差分析。
6. **科研方向**：持续学习、ViT 量化、2D/3D 分割、3D Gaussian Splatting，分别建立项目环境。

---

# 15. 每个知识点的标准产物

至少留下：

1. 最小可运行代码；
2. 输入输出形状说明；
3. 一个失败或边界案例；
4. 一句机制结论。

模板：

```markdown
## torch.stack

### 目的
沿新维度拼接多个同形状 Tensor。

### Shape Contract
输入：N 个形状均为 `(B, C)` 的 Tensor。
输出：若 `dim=0`，形状为 `(N, B, C)`。

### 最小代码
...

### 易错点
`stack` 新增维度，`cat` 在已有维度拼接。

### 验证结论
...
```

---

# 16. 日常学习闭环

一次 60～120 分钟学习可按以下比例：

```text
10% 阅读概念与文档
35% 最小代码验证
25% 修改输入、制造异常、验证边界
20% 整理为可复现脚本
10% 记录结论与未解决问题
```

遇到不理解的代码：

```text
缩小输入
→ 打印形状
→ 手算一个例子
→ 移除非核心模块
→ 验证梯度
→ 恢复完整代码
```

---

# 17. 最终决策

- 几行代码：**IPython**；
- 需要保存：**短 `.py`**；
- 模块学习：**`.py` + `pytest`**；
- 数据分析：**Jupyter Notebook**；
- 完整训练：**训练脚本 + 配置 + TensorBoard + Checkpoint**；
- 正式科研：**独立 Conda 环境 + 标准项目结构 + Git + 服务器**。

> **交互环境用于快速提出问题，脚本用于稳定回答问题，测试用于长期保存答案，训练项目用于产生可复现实验结论。**

# ZO 优化器扫描指南

## 概述

此脚本用于对比不同优化器在 ZO（Zeroth-Order）和 FO（First-Order/BP）模式下的性能。

## 当前配置

| 参数 | 值 |
|------|-----|
| **模式** | ZO（Zero-Order） + FO（BP 基准） |
| **批次大小** | 2 |
| **学习率** | 1e-6 |
| **查询预算（q）** | 1 |
| **优化器** | adam, muon |
| **训练轮数** | 10 |
| **GPU** | 2,3 |
| **最大并行任务** | 2 |

## 快速开始

### 方式 1：运行预设脚本（推荐）

```bash
cd /data/cdq/current_project/zo-test
bash run_zo_optimizer_sweep.sh
```

这将运行以下实验组合：
- **ZO + Adam**
- **ZO + Muon**
- **FO + Adam**（BP 基准）
- **FO + Muon**（BP 基准）

### 方式 2：手动配置 parallel_sweep.sh

如果需要自定义参数，直接使用 `parallel_sweep.sh`：

```bash
bash parallel_sweep.sh \
    --modes "ZO,FO" \
    --scopes "full" \
    --batch-sizes "2" \
    --query-budgets "1" \
    --learning-rates "1e-6" \
    --optimizers "adam,muon" \
    --epochs 10 \
    --parallel 2 \
    --gpus "2,3"
```

## 参数说明

### parallel_sweep.sh 支持的参数

```
--modes              优化模式：ZO 或 FO（默认：ZO,FO）
--scopes             训练范围：full 或 reduced（默认：full）
--batch-sizes        批次大小列表（默认：2）
--query-budgets      查询预算 q 列表，仅用于 ZO（默认：1）
--learning-rates     学习率列表（默认：1e-6）
--optimizers         优化器列表：adam, muon（默认：adam,muon）
--epochs             训练轮数（默认：10）
--parallel           最大并行任务数（默认：2）
--gpus               GPU ID 列表，逗号分隔（默认：2,3）
--log-interval       CSV 日志记录间隔（默认：10 步）
```

## 实验输出

运行完成后，结果将保存在以下位置：

```
results/                          # 损失曲线图
├── ZO_full_bs2_q1_lr1e-6_optadam.png
├── ZO_full_bs2_q1_lr1e-6_optmuon.png
├── FO_full_bs2_q1_lr1e-6_optadam.png
└── FO_full_bs2_q1_lr1e-6_optmuon.png

csv_logs_mezo_epochs10_large/     # 详细训练日志
├── ZO_full_bs2_q1_lr1e-6_optadam.csv
├── ZO_full_bs2_q1_lr1e-6_optmuon.csv
├── FO_full_bs2_q1_lr1e-6_optadam.csv
└── FO_full_bs2_q1_lr1e-6_optmuon.csv

job_logs_*/                       # 各任务的详细日志
```

## CSV 日志格式

每个 CSV 文件包含以下列：

| 列名 | 说明 |
|------|------|
| timestamp | 记录时间戳 |
| epoch | 训练轮次 |
| step | 训练步数 |
| mode | 优化模式（ZO/FO） |
| scope | 训练范围（full/reduced） |
| q | 查询预算（ZO 使用）|
| lr | 学习率 |
| batch_size | 批次大小 |
| loss | 交叉熵损失 |
| grad_norm | 梯度范数 |

## 高级用法示例

### 示例 1：测试多个学习率

```bash
bash parallel_sweep.sh \
    --modes "ZO" \
    --batch-sizes "2" \
    --query-budgets "1" \
    --learning-rates "1e-7,1e-6,1e-5" \
    --optimizers "adam,muon" \
    --epochs 5
```

### 示例 2：测试不同的查询预算

```bash
bash parallel_sweep.sh \
    --modes "ZO" \
    --batch-sizes "2" \
    --query-budgets "1,2,4,8" \
    --learning-rates "1e-6" \
    --optimizers "adam" \
    --epochs 5
```

### 示例 3：仅运行 FO 基准测试

```bash
bash parallel_sweep.sh \
    --modes "FO" \
    --scopes "full" \
    --batch-sizes "2" \
    --learning-rates "1e-6" \
    --optimizers "adam,muon" \
    --epochs 10
```

## 性能监控

运行中可以监控：

```bash
# 查看 GPU 使用情况
watch -n 1 nvidia-smi

# 查看进程和内存
watch -n 1 ps aux | grep python
```

## 结果分析

### 生成对比图表

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV
zo_adam = pd.read_csv('csv_logs_mezo_epochs10_large/ZO_full_bs2_q1_lr1e-6_optadam.csv')
zo_muon = pd.read_csv('csv_logs_mezo_epochs10_large/ZO_full_bs2_q1_lr1e-6_optmuon.csv')
fo_adam = pd.read_csv('csv_logs_mezo_epochs10_large/FO_full_bs2_q1_lr1e-6_optadam.csv')
fo_muon = pd.read_csv('csv_logs_mezo_epochs10_large/FO_full_bs2_q1_lr1e-6_optmuon.csv')

# 绘制对比图
plt.figure(figsize=(12, 6))
plt.plot(zo_adam['step'], zo_adam['loss'], label='ZO + Adam')
plt.plot(zo_muon['step'], zo_muon['loss'], label='ZO + Muon')
plt.plot(fo_adam['step'], fo_adam['loss'], label='FO + Adam')
plt.plot(fo_muon['step'], fo_muon['loss'], label='FO + Muon')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('optimizer_comparison.png')
plt.show()
```

## 故障排除

### 问题：GPU 内存不足

**解决方案**：
1. 减少 `--parallel` 值
2. 减少 `--batch-sizes`
3. 使用 `--query-budgets "1"`（最小值）

### 问题：运行速度过慢

**解决方案**：
1. 增加 `--parallel` 值（如果有多个 GPU）
2. 减少 `--epochs`
3. 减少学习率数量

### 问题：某些任务失败

**解决方案**：
1. 检查 `job_logs_*/` 目录中的错误日志
2. 确保有足够的磁盘空间
3. 确保所有依赖包已安装

## 注意事项

- **ZO vs FO**：ZO 模式需要多次前向传播（由 q 决定），所以会较慢
- **Optimizer 支持**：当前支持 `adam` 和 `muon`，`bp` 是通过 FO 模式实现的
- **并行化**：脚本自动轮换 GPU 分配，确保负载均衡
- **日志记录**：每个任务的详细日志在 `job_logs_*` 目录中

## 联系与反馈

如有问题或建议，请检查 `parallel_sweep_*.log` 文件获取详细信息。

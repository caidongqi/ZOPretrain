# ZO 优化器扫描 - 设置总结

## ✅ 完成的工作

已成功配置了一套完整的参数扫描系统，用于对比不同优化器在 ZO（Zeroth-Order）和 FO（First-Order/BP）模式下的性能。

## 📋 修改的文件

### 1. `parallel_sweep.sh` (主扫描脚本)
**变更内容：**
- ✅ 添加 `OPTIMIZERS` 数组支持
- ✅ 添加 `--optimizers` 命令行参数
- ✅ 修改实验生成逻辑，支持优化器维度扫描
- ✅ 修改 `run_single_experiment()` 函数，传递 `--optimizer` 参数

**支持的参数：**
```bash
--optimizers 'adam,muon'     # 可选：adam, muon
--modes 'ZO,FO'              # 可选：ZO, FO
--batch-sizes '2'            # 批次大小
--query-budgets '1'          # ZO 查询预算
--learning-rates '1e-6'      # 学习率
--epochs 10                  # 训练轮数
--parallel 2                 # 最大并行任务数
--gpus '2,3'                 # GPU ID 列表
```

### 2. 新增脚本：`run_zo_optimizer_sweep.sh`
**功能：**
- 🚀 一键运行完整的优化器扫描
- 📊 包含 ZO + Adam、ZO + Muon、FO + Adam、FO + Muon 四个实验
- ⚙️ 预配置了推荐的参数：bs=2, lr=1e-6, q=1, epochs=10

**使用方法：**
```bash
bash run_zo_optimizer_sweep.sh
```

### 3. 新增脚本：`test_optimizer_sweep.sh`
**功能：**
- 🧪 快速测试配置是否正确
- ⚡ 运行单个 epoch 进行验证
- 📋 生成一份完整的示例输出

**使用方法：**
```bash
bash test_optimizer_sweep.sh
```

### 4. 新增文档：`OPTIMIZER_SWEEP_README.md`
**包含内容：**
- 📖 详细的使用指南
- 🔧 参数说明
- 📊 输出格式说明
- 💡 高级用法示例
- 🐛 故障排除指南
- 📈 结果分析示例

## 🎯 实验设计

### 默认配置
| 维度 | 值 |
|------|-----|
| **模式** | ZO + FO（BP 基准） |
| **优化器** | Adam、Muon |
| **批次大小** | 2 |
| **学习率** | 1e-6 |
| **查询预算（q）** | 1 |
| **训练轮数** | 10 |
| **GPU** | 2,3（可配置） |

### 生成的实验组合
```
1. ZO + Adam       (bs=2, lr=1e-6, q=1)
2. ZO + Muon       (bs=2, lr=1e-6, q=1)
3. FO + Adam       (bs=2, lr=1e-6)  ← BP 基准
4. FO + Muon       (bs=2, lr=1e-6)  ← BP 基准
```

## 🚀 快速开始

### 步骤 1：测试配置（可选但推荐）
```bash
cd /data/cdq/current_project/zo-test
bash test_optimizer_sweep.sh
```

### 步骤 2：运行完整扫描
```bash
bash run_zo_optimizer_sweep.sh
```

### 步骤 3：检查结果
```bash
# 查看损失曲线
ls -la results/*.png

# 查看训练日志
ls -la csv_logs_mezo_epochs10_large/*.csv

# 查看详细日志
ls -la job_logs_*/*.log
```

## 📊 输出文件

### 损失曲线 (PNG)
```
results/
├── ZO_full_bs2_q1_lr1e-6_optadam.png
├── ZO_full_bs2_q1_lr1e-6_optmuon.png
├── FO_full_bs2_q1_lr1e-6_optadam.png
└── FO_full_bs2_q1_lr1e-6_optmuon.png
```

### 训练日志 (CSV)
```
csv_logs_mezo_epochs10_large/
├── ZO_full_bs2_q1_lr1e-6_optadam.csv
├── ZO_full_bs2_q1_lr1e-6_optmuon.csv
├── FO_full_bs2_q1_lr1e-6_optadam.csv
└── FO_full_bs2_q1_lr1e-6_optmuon.csv
```

### 详细日志 (TXT)
```
job_logs_TIMESTAMP/
├── ZO_full_bs2_q1_lr1e-6_optadam.log
├── ZO_full_bs2_q1_lr1e-6_optmuon.log
├── FO_full_bs2_q1_lr1e-6_optadam.log
└── FO_full_bs2_q1_lr1e-6_optmuon.log
```

## 🔧 高级配置

### 自定义学习率扫描
```bash
bash parallel_sweep.sh \
    --modes "ZO" \
    --batch-sizes "2" \
    --query-budgets "1" \
    --learning-rates "1e-7,1e-6,1e-5" \
    --optimizers "adam,muon" \
    --epochs 5
```

### 自定义查询预算扫描
```bash
bash parallel_sweep.sh \
    --modes "ZO" \
    --batch-sizes "2" \
    --query-budgets "1,2,4,8" \
    --learning-rates "1e-6" \
    --optimizers "adam" \
    --epochs 5
```

### 仅运行 FO 基准
```bash
bash parallel_sweep.sh \
    --modes "FO" \
    --batch-sizes "2" \
    --learning-rates "1e-6" \
    --optimizers "adam,muon" \
    --epochs 10
```

## 📈 数据分析示例

### Python 对比分析脚本
```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
zo_adam = pd.read_csv('csv_logs_mezo_epochs10_large/ZO_full_bs2_q1_lr1e-6_optadam.csv')
zo_muon = pd.read_csv('csv_logs_mezo_epochs10_large/ZO_full_bs2_q1_lr1e-6_optmuon.csv')
fo_adam = pd.read_csv('csv_logs_mezo_epochs10_large/FO_full_bs2_q1_lr1e-6_optadam.csv')
fo_muon = pd.read_csv('csv_logs_mezo_epochs10_large/FO_full_bs2_q1_lr1e-6_optmuon.csv')

# 绘制对比
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(zo_adam['step'], zo_adam['loss'], label='ZO + Adam')
axes[0, 0].set_title('ZO + Adam')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid()

axes[0, 1].plot(zo_muon['step'], zo_muon['loss'], label='ZO + Muon')
axes[0, 1].set_title('ZO + Muon')
axes[0, 1].grid()

axes[1, 0].plot(fo_adam['step'], fo_adam['loss'], label='FO + Adam (BP)')
axes[1, 0].set_title('FO + Adam (BP)')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid()

axes[1, 1].plot(fo_muon['step'], fo_muon['loss'], label='FO + Muon (BP)')
axes[1, 1].set_title('FO + Muon (BP)')
axes[1, 1].grid()

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150)
plt.show()
```

## 🎓 理解实验

### 什么是 ZO（Zeroth-Order）？
- 只使用损失值，无需计算梯度
- 更多的前向传播（由 q 控制）
- 更少的反向传播
- 适合大模型训练

### 什么是 FO（First-Order）？
- 使用标准反向传播计算梯度
- 这是通常的 BP（Backpropagation）
- 更少的前向传播
- 作为基准进行对比

### Adam vs Muon 优化器
- **Adam**：经典的自适应学习率优化器
- **Muon**：新型优化器，声称在大模型上性能更好

## ⚠️ 注意事项

1. **GPU 内存**：ZO 模式需要多个前向传播，可能需要较多 GPU 内存
2. **计算时间**：ZO 比 FO 慢（因为需要 q 倍的前向传播）
3. **精度**：q=1 是最小的查询预算，可增加 q 以获得更好的梯度估计
4. **并行化**：脚本自动在 GPU 2,3 上轮换任务

## 🐛 故障排除

### GPU 内存不足
```bash
# 减少并行任务数
bash parallel_sweep.sh --parallel 1 ...

# 或者只在单个 GPU 上运行
bash parallel_sweep.sh --gpus "2" ...
```

### 任务失败
```bash
# 查看详细错误日志
cat job_logs_TIMESTAMP/ZO_full_bs2_q1_lr1e-6_optadam.log
```

### 运行过慢
```bash
# 减少 epoch 数进行测试
bash test_optimizer_sweep.sh  # 只运行 1 个 epoch
```

## 📝 文件变更清单

- ✅ `parallel_sweep.sh` - 添加优化器支持
- ✅ `run_zo_optimizer_sweep.sh` - 新增（一键启动脚本）
- ✅ `test_optimizer_sweep.sh` - 新增（测试脚本）
- ✅ `OPTIMIZER_SWEEP_README.md` - 新增（详细文档）
- ✅ `SETUP_SUMMARY.md` - 新增（本文件）

## 🎯 后续步骤

1. ✅ 运行 `test_optimizer_sweep.sh` 验证设置
2. ✅ 运行 `run_zo_optimizer_sweep.sh` 执行完整扫描
3. ✅ 查看 `results/` 和 `csv_logs_mezo_epochs10_large/` 中的结果
4. ✅ 使用 Python 脚本进行数据分析和可视化

---

**创建日期：** 2025-10-29  
**脚本版本：** 1.0  
**优化器支持：** Adam, Muon  
**模式支持：** ZO, FO (BP)

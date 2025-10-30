# 🚀 快速开始指南

## 一句话总结

扫描 ZO 和 FO（BP）在 Adam、Muon 优化器上的表现，配置为 bs=2, lr=1e-6, q=1。

## ⚡ 最快上手（3 步）

### 1️⃣ 进入项目目录
```bash
cd /data/cdq/current_project/zo-test
```

### 2️⃣ 运行完整扫描
```bash
bash run_zo_optimizer_sweep.sh
```

### 3️⃣ 查看结果
```bash
# 查看损失曲线
ls results/*.png

# 查看训练数据
ls csv_logs_mezo_epochs10_large/*.csv
```

## 📋 默认配置

| 参数 | 值 |
|------|-----|
| Mode | ZO + FO |
| Optimizers | Adam, Muon |
| Batch Size | 2 |
| Learning Rate | 1e-6 |
| Query Budget (q) | 1 |
| Epochs | 10 |

## 📊 会生成的实验

```
✅ ZO + Adam        → Loss curve + CSV
✅ ZO + Muon        → Loss curve + CSV
✅ FO + Adam (BP)   → Loss curve + CSV
✅ FO + Muon (BP)   → Loss curve + CSV
```

## 🧪 先测试一下（推荐）

运行测试版本，只需 1 分钟左右：

```bash
bash test_optimizer_sweep.sh
```

## 🎯 自定义参数

### 只测试 Adam
```bash
bash parallel_sweep.sh \
    --modes "ZO,FO" \
    --optimizers "adam" \
    --batch-sizes "2" \
    --query-budgets "1" \
    --learning-rates "1e-6" \
    --epochs 5
```

### 测试多个学习率
```bash
bash parallel_sweep.sh \
    --learning-rates "1e-7,1e-6,1e-5" \
    --optimizers "adam,muon"
```

### 测试多个查询预算（仅 ZO）
```bash
bash parallel_sweep.sh \
    --modes "ZO" \
    --query-budgets "1,2,4,8" \
    --optimizers "adam"
```

## 📊 输出位置

```
results/                              # PNG 图表
csv_logs_mezo_epochs10_large/         # CSV 数据
job_logs_YYYYMMDD_HHMMSS/            # 详细日志
parallel_sweep_YYYYMMDD_HHMMSS.log   # 运行日志
```

## 🔗 相关文件

| 文件 | 用途 |
|------|------|
| `parallel_sweep.sh` | 主脚本（支持所有参数） |
| `run_zo_optimizer_sweep.sh` | 一键启动（推荐） |
| `test_optimizer_sweep.sh` | 快速测试 |
| `SETUP_SUMMARY.md` | 完整设置说明 |
| `OPTIMIZER_SWEEP_README.md` | 详细文档 |

## 💡 常见命令

```bash
# 检查 GPU 使用
watch -n 1 nvidia-smi

# 实时查看日志
tail -f parallel_sweep_*.log

# 查看某个任务的错误
cat job_logs_*/ZO_full_bs2_q1_lr1e-6_optadam.log
```

## ⚙️ 所有参数一览

```bash
bash parallel_sweep.sh \
    --modes "ZO,FO" \                    # 优化模式
    --scopes "full" \                    # 训练范围
    --batch-sizes "2" \                  # 批次大小
    --query-budgets "1" \                # ZO 查询预算
    --learning-rates "1e-6" \            # 学习率
    --optimizers "adam,muon" \           # 优化器
    --epochs 10 \                        # 训练轮数
    --parallel 2 \                       # 并行任务数
    --gpus "2,3" \                       # GPU ID
    --log-interval 10                    # 日志间隔
```

## 🎓 关键概念

- **ZO（Zeroth-Order）**：只用损失值，无梯度反向传播
- **FO（First-Order）**：标准反向传播（BP）
- **q（Query Budget）**：ZO 的随机方向个数
- **Adam**：经典优化器
- **Muon**：新型优化器

## ❓ 常见问题

**Q：运行多久？**
A：10 epochs，通常 30-60 分钟（取决于 GPU）

**Q：需要多少 GPU 内存？**
A：ZO 需要的内存更多（多个前向传播）

**Q：能否使用单 GPU？**
A：可以，改为 `--gpus "2"` 或 `--gpus "3"`

**Q：如何减速测试？**
A：运行 `bash test_optimizer_sweep.sh`，仅需 1 epoch

---

**更多信息：** 见 `OPTIMIZER_SWEEP_README.md`

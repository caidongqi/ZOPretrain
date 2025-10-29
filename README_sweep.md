# ZO vs FO Parameter Sweep

这个项目实现了零阶优化(Zero-Order)和一阶优化(First-Order)方法的对比实验，支持参数扫描和实时日志记录。

## 🚀 快速开始

### 1. 快速测试
```bash
./quick_test.sh
```
这将运行3个快速实验：
- FO (First-Order) 方法
- ZO (q=1) 方法  
- ZO (q=4) 方法

### 2. 完整参数扫描
```bash
./sweep.sh
```
这将运行所有参数组合的实验。

## 📊 实验配置

### 参数范围
- **优化方法**: FO, ZO
- **训练范围**: reduced (最后一层)
- **Batch Size**: 4, 8, 16
- **Query Budget (ZO)**: 1, 2, 4, 8
- **学习率**:
  - FO: 1e-4, 5e-5
  - ZO: 1e-5, 5e-6
- **训练轮数**: 2
- **日志间隔**: 每5步记录一次

### 总实验数
- FO实验: 2 × 3 = 6个
- ZO实验: 2 × 3 × 4 = 24个
- **总计**: 30个实验

## 📁 输出文件

### 目录结构
```
results/          # PNG图表文件
csv_logs/         # CSV日志文件
cache/            # 数据集缓存
sweep_*.log       # 运行日志
sweep_summary_*.txt  # 实验总结
```

### CSV日志格式
每行包含以下字段：
- `timestamp`: 时间戳
- `epoch`: 训练轮数
- `step`: 训练步数
- `mode`: 优化方法 (FO/ZO)
- `scope`: 训练范围
- `q`: Query budget (ZO) 或 N/A (FO)
- `lr`: 学习率
- `batch_size`: 批次大小
- `loss`: 当前损失值
- `grad_norm`: 梯度范数 (ZO)

## 🔧 自定义实验

### 修改参数
编辑 `sweep.sh` 中的配置变量：
```bash
MODES=("FO" "ZO")
BATCH_SIZES=(4 8 16)
QUERY_BUDGETS=(1 2 4 8)
LEARNING_RATES_FO=(1e-4 5e-5)
LEARNING_RATES_ZO=(1e-5 5e-6)
```

### 单个实验
```bash
python reproduce_zo_paper.py \
    --mode ZO \
    --scope reduced \
    --batch_size 8 \
    --query_budget_q 4 \
    --learning_rate 1e-5 \
    --epochs 2 \
    --csv_file my_experiment.csv \
    --log_interval 10
```

## 📈 结果分析

### 1. 查看PNG图表
```bash
ls results/*.png
```

### 2. 分析CSV数据
```bash
# 查看所有实验的最终损失
tail -n +2 csv_logs/*.csv | cut -d',' -f1,3,4,5,6,7,8,9 | sort
```

### 3. 生成性能报告
`sweep.sh` 会自动生成性能总结，包括：
- 实验成功率
- 最终损失值
- 平均损失值
- 运行时间统计

## 🐛 故障排除

### 常见问题
1. **内存不足**: 减少batch_size或使用更小的模型
2. **CUDA错误**: 检查GPU内存，考虑使用CPU
3. **数据集加载慢**: 首次运行会创建缓存，后续会更快

### 调试模式
```bash
# 查看详细日志
tail -f sweep_*.log

# 运行单个实验进行调试
python reproduce_zo_paper.py --mode ZO --scope reduced --batch_size 4 --query_budget_q 1 --learning_rate 1e-5 --epochs 1
```

## 📋 实验说明

### ZO梯度估计
- 每个训练步骤使用q个随机方向
- 每个方向需要2次前向传播 (+εu 和 -εu)
- 总前向传播次数: 2 × q

### 数据集缓存
- 首次运行会下载并缓存数据集
- 缓存文件保存在 `cache/` 目录
- 后续运行会直接加载缓存，大幅提升速度

### 实时监控
- CSV文件每N步更新一次
- 可以实时监控训练进度
- 支持中断后继续分析已有数据

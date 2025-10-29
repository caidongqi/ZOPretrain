# 实验结果可视化

这个项目现在支持自动绘制所有实验结果的loss曲线，并提供多种分析视角。

## 🚀 快速开始

### 1. 一键运行并绘图
```bash
./run_and_plot.sh
```
这会自动运行并行实验并生成所有图表。

### 2. 快速绘图（如果已有CSV数据）
```bash
python quick_plot.py
```

### 3. 详细分析绘图
```bash
python plot_all_results.py --all
```

## 📊 绘图功能

### 快速绘图 (`quick_plot.py`)
- **概览图**: 4个子图显示所有实验的loss曲线
- **按模式分组**: FO vs ZO对比
- **按scope分组**: Reduced vs Full对比
- **按batch size分组**: 不同批次大小的对比
- **统计信息**: 显示最佳实验和基本统计

### 详细分析 (`plot_all_results.py`)
- **概览图**: 所有实验的loss曲线对比
- **ZO分析图**: 专门分析ZO实验的6个子图
- **FO vs ZO对比图**: 两种方法的详细对比
- **总结报告**: 生成详细的文本报告

## 🎯 使用方法

### 基本用法
```bash
# 生成所有图表
python plot_all_results.py --all

# 只生成概览图
python plot_all_results.py --overview

# 只生成ZO分析图
python plot_all_results.py --zo-analysis

# 只生成FO vs ZO对比图
python plot_all_results.py --comparison

# 只生成总结报告
python plot_all_results.py --summary
```

### 自定义参数
```bash
python plot_all_results.py \
    --csv-dir csv_logs \
    --output-dir my_plots \
    --figsize 20 15 \
    --all
```

## 📁 输出文件

### 快速绘图输出
- `plots/all_loss_curves.png`: 4个子图的概览

### 详细分析输出
- `plots/loss_curves_overview.png`: 概览图
- `plots/zo_analysis.png`: ZO分析图
- `plots/fo_vs_zo_comparison.png`: FO vs ZO对比图
- `plots/experiment_summary.txt`: 总结报告

## 🔍 图表说明

### 概览图 (4个子图)
1. **所有实验**: 显示每个实验的loss曲线
2. **按模式分组**: FO和ZO方法的对比
3. **按scope分组**: Reduced和Full模型的对比
4. **按batch size分组**: 不同批次大小的对比

### ZO分析图 (6个子图)
1. **Loss by Query Budget**: 不同q值的loss曲线
2. **Loss by Batch Size**: 不同批次大小的loss曲线
3. **Loss by Scope**: 不同scope的loss曲线
4. **Final Loss vs Query Budget**: 最终loss与q值的关系
5. **Gradient Norm Evolution**: 梯度范数变化
6. **Loss Reduction Percentage**: 收敛速度分析

### FO vs ZO对比图 (4个子图)
1. **Average Loss Comparison**: 平均loss曲线对比
2. **Final Loss Distribution**: 最终loss分布对比
3. **Convergence Speed Comparison**: 收敛速度对比
4. **Training Stability**: 训练稳定性对比

## 📈 分析指标

### 基本指标
- **最终loss**: 每个实验的最终loss值
- **平均loss**: 整个训练过程的平均loss
- **loss下降率**: (初始loss - 最终loss) / 初始loss × 100%
- **梯度范数**: ZO方法的梯度估计范数

### 对比指标
- **收敛速度**: loss下降的速率
- **训练稳定性**: loss的方差和标准差
- **最终性能**: 不同方法/参数的最优结果
- **计算效率**: 不同q值的计算成本

## 🛠️ 自定义分析

### 修改绘图参数
```python
# 在plot_all_results.py中修改
figsize=(20, 15)  # 图片大小
dpi=300           # 图片分辨率
alpha=0.7         # 线条透明度
linewidth=2       # 线条宽度
```

### 添加新的分析维度
```python
# 在plot_all_results.py中添加新的子图
def plot_custom_analysis(df, output_dir="plots"):
    # 自定义分析逻辑
    pass
```

## 🔧 故障排除

### 常见问题
1. **没有CSV文件**: 确保实验已成功运行
2. **图片显示问题**: 检查matplotlib后端设置
3. **中文显示问题**: 确保系统有中文字体

### 调试模式
```bash
# 检查CSV文件
ls -la csv_logs/*.csv

# 检查数据内容
head -5 csv_logs/*.csv

# 检查图片生成
ls -la plots/*.png
```

## 📋 工作流程

1. **运行实验**: 使用`./parallel_sweep.sh`或`./run_and_plot.sh`
2. **检查数据**: 确认CSV文件已生成
3. **生成图表**: 运行绘图脚本
4. **分析结果**: 查看生成的图片和报告
5. **调整参数**: 根据结果调整实验参数
6. **重新运行**: 迭代优化实验

现在您可以轻松地可视化和分析所有实验结果了！


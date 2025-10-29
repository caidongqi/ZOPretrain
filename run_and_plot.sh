#!/bin/bash

# 运行并行实验并自动绘图
# Run parallel experiments and automatically plot results

set -e

echo "🚀 运行并行实验并自动绘图"
echo "=========================="

# 运行并行实验
echo "📊 开始运行并行实验..."
./parallel_sweep.sh

# 等待所有实验完成
echo "⏳ 等待所有实验完成..."
wait

# 检查是否有CSV文件生成
if [ ! -d "csv_logs" ] || [ -z "$(ls -A csv_logs/*.csv 2>/dev/null)" ]; then
    echo "❌ 没有找到CSV文件，请检查实验是否成功运行"
    exit 1
fi

echo "✅ 实验完成，开始绘图..."

# 运行快速绘图
echo "📈 生成loss曲线图..."
python quick_plot.py

# 运行详细分析（如果存在）
if [ -f "plot_all_results.py" ]; then
    echo "📊 生成详细分析图..."
    python plot_all_results.py --all
fi

echo ""
echo "🎉 完成！"
echo "📁 查看结果:"
echo "  - 图片: plots/"
echo "  - CSV数据: csv_logs/"
echo "  - 实验日志: job_logs_*/"
echo ""
echo "🔍 快速查看图片:"
echo "  ls plots/*.png"


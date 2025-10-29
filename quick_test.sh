#!/bin/bash

# Quick test script for ZO vs FO comparison
# 快速测试脚本，用于验证功能

set -e

echo "🚀 Quick Test: ZO vs FO Comparison"
echo "=================================="

# 创建目录
mkdir -p results csv_logs cache

# 测试参数
EPOCHS=1
LOG_INTERVAL=5

echo "📊 Running FO experiment..."
python reproduce_zo_paper.py \
    --mode FO \
    --scope reduced \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --epochs $EPOCHS \
    --csv_file csv_logs/fo_test.csv \
    --log_interval $LOG_INTERVAL

echo ""
echo "📊 Running ZO experiment (q=1)..."
python reproduce_zo_paper.py \
    --mode ZO \
    --scope reduced \
    --batch_size 4 \
    --query_budget_q 1 \
    --learning_rate 1e-5 \
    --epochs $EPOCHS \
    --csv_file csv_logs/zo_q1_test.csv \
    --log_interval $LOG_INTERVAL

echo ""
echo "📊 Running ZO experiment (q=4)..."
python reproduce_zo_paper.py \
    --mode ZO \
    --scope reduced \
    --batch_size 4 \
    --query_budget_q 4 \
    --learning_rate 1e-5 \
    --epochs $EPOCHS \
    --csv_file csv_logs/zo_q4_test.csv \
    --log_interval $LOG_INTERVAL

echo ""
echo "✅ Quick test completed!"
echo "Check results in:"
echo "  - PNG plots: results/"
echo "  - CSV logs: csv_logs/"
echo "  - Dataset cache: cache/"

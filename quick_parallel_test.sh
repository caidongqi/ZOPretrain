#!/bin/bash

# Quick parallel test script
# 快速并行测试脚本

set -e

echo "🚀 Quick Parallel Test: ZO vs FO"
echo "================================"

# 创建目录
mkdir -p results csv_logs cache job_logs

# 测试参数
EPOCHS=1
LOG_INTERVAL=5
MAX_PARALLEL=2

# 检测GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ $GPU_COUNT -gt 0 ]; then
        echo "🔍 Detected $GPU_COUNT GPU(s)"
        GPU_IDS="0"
        if [ $GPU_COUNT -gt 1 ]; then
            GPU_IDS="0,1"
        fi
    else
        echo "⚠️  No GPUs detected, using CPU"
        GPU_IDS="cpu"
    fi
else
    echo "⚠️  nvidia-smi not found, using CPU"
    GPU_IDS="cpu"
fi

echo "GPU IDs: $GPU_IDS"
echo "Max parallel jobs: $MAX_PARALLEL"
echo ""

# 运行并行测试
./parallel_sweep.sh \
    --parallel $MAX_PARALLEL \
    --gpus "$GPU_IDS" \
    --modes "FO,ZO" \
    --scopes "reduced" \
    --batch-sizes "2,4" \
    --query-budgets "1,2" \
    --learning-rates "1e-4,1e-5" \
    --epochs $EPOCHS \
    --log-interval $LOG_INTERVAL

echo ""
echo "✅ Quick parallel test completed!"
echo "Check results in:"
echo "  - PNG plots: results/"
echo "  - CSV logs: csv_logs/"
echo "  - Job logs: job_logs/"


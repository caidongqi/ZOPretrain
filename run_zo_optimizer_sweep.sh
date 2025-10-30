#!/bin/bash

# ZO 优化器扫描脚本
# 配置：bs=2, learning_rate=1e-6, q=1
# 优化器：adam, muon, bp

set -e

# 颜色定义
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting ZO Optimizer Sweep${NC}"
echo -e "${BLUE}==============================${NC}"
echo "Configuration:"
echo "  - Mode: ZO (Zeroth-Order)"
echo "  - Batch Size: 2"
echo "  - Learning Rate: 1e-6"
echo "  - Query Budget (q): 1"
echo "  - Optimizers: adam, muon, bp (FO baseline)"
echo ""

# 运行 parallel_sweep.sh，指定参数
bash ./parallel_sweep.sh \
    --modes "ZO,FO" \
    --scopes "full" \
    --batch-sizes "2" \
    --query-budgets "1" \
    --learning-rates "1e-6" \
    --optimizers "sgd" \
    --epochs 10 \
    --parallel 2 \
    --gpus "2,3"

echo -e "${GREEN}✅ ZO Optimizer Sweep completed!${NC}"

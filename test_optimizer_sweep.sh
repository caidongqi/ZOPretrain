#!/bin/bash

# 测试脚本：验证优化器扫描配置
# 运行单个 epoch 来快速验证配置

set -e

# 颜色定义
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🧪 Testing ZO Optimizer Sweep Configuration${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo "This test runs a quick validation with 1 epoch on a single GPU."
echo ""

# 运行测试：单 epoch，单个 GPU，最小配置
bash ./parallel_sweep.sh \
    --modes "ZO,FO" \
    --scopes "full" \
    --batch-sizes "2" \
    --query-budgets "1" \
    --learning-rates "1e-6" \
    --optimizers "adam,muon" \
    --epochs 1 \
    --parallel 1 \
    --gpus "2"

echo ""
echo -e "${GREEN}✅ Test completed successfully!${NC}"
echo ""
echo "Generated files:"
echo "  - Loss curves: results/*.png"
echo "  - Training logs: csv_logs_mezo_epochs10_large/*.csv"
echo "  - Job logs: job_logs_*/*.log"
echo ""
echo -e "${YELLOW}📌 Next step: Run full sweep with:${NC}"
echo "  bash run_zo_optimizer_sweep.sh"

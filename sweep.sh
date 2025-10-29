#!/bin/bash

# ZO vs FO Parameter Sweep Script
# 测试FO和ZO在不同batch size和query budget下的结果

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置参数
MODES=("ZO")
SCOPES=("reduced" "full")  # 可以添加 "full" 如果需要
BATCH_SIZES=(1 2 4)
QUERY_BUDGETS=(1 2 4 8)
# LEARNING_RATES_FO=(1e-4 5e-5)
LEARNING_RATES_ZO=(3e-4)
EPOCHS=1
LOG_INTERVAL=10

# 创建结果目录
RESULTS_DIR="results"
CSV_DIR="csv_logs_mezo"
CACHE_DIR="cache"
mkdir -p "$RESULTS_DIR" "$CSV_DIR" "$CACHE_DIR"

# 日志文件
LOG_FILE="sweep_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_FILE="sweep_summary_$(date +%Y%m%d_%H%M%S).txt"

echo -e "${BLUE}🚀 Starting ZO vs FO Parameter Sweep${NC}"
echo -e "${BLUE}====================================${NC}"
echo "Results will be saved to: $RESULTS_DIR"
echo "CSV logs will be saved to: $CSV_DIR"
echo "Dataset cache: $CACHE_DIR"
echo "Log file: $LOG_FILE"
echo ""

# 计数器
total_experiments=0
successful_experiments=0
failed_experiments=0

# 记录开始时间
start_time=$(date +%s)

# 函数：运行单个实验
run_experiment() {
    local mode=$1
    local scope=$2
    local batch_size=$3
    local q=$4
    local lr=$5
    local epochs=$6
    
    total_experiments=$((total_experiments + 1))
    
    # 生成实验ID
    local exp_id="${mode}_${scope}_bs${batch_size}_q${q}_lr${lr}"
    local csv_file="${CSV_DIR}/${exp_id}.csv"
    
    echo -e "${YELLOW}📊 Experiment $total_experiments: $exp_id${NC}"
    echo "Mode: $mode, Scope: $scope, Batch Size: $batch_size, Q: $q, LR: $lr"
    
    # 构建命令
    local cmd="python reproduce_zo_paper.py"
    cmd="$cmd --mode $mode"
    cmd="$cmd --scope $scope"
    cmd="$cmd --batch_size $batch_size"
    cmd="$cmd --learning_rate $lr"
    cmd="$cmd --epochs $epochs"
    cmd="$cmd --csv_file $csv_file"
    cmd="$cmd --log_interval $LOG_INTERVAL"
    
    if [ "$mode" = "ZO" ]; then
        cmd="$cmd --query_budget_q $q"
    fi
    
    echo "Command: $cmd"
    echo "----------------------------------------"
    
    # 运行实验
    if eval $cmd >> "$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✅ Experiment $exp_id completed successfully${NC}"
        successful_experiments=$((successful_experiments + 1))
        
        # 记录到总结文件
        echo "$exp_id,SUCCESS" >> "$SUMMARY_FILE"
    else
        echo -e "${RED}❌ Experiment $exp_id failed${NC}"
        failed_experiments=$((failed_experiments + 1))
        
        # 记录到总结文件
        echo "$exp_id,FAILED" >> "$SUMMARY_FILE"
    fi
    
    echo ""
}

# 函数：生成最终报告
generate_report() {
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo -e "${BLUE}📋 SWEEP SUMMARY REPORT${NC}"
    echo -e "${BLUE}======================${NC}"
    echo "Total experiments: $total_experiments"
    echo -e "Successful: ${GREEN}$successful_experiments${NC}"
    echo -e "Failed: ${RED}$failed_experiments${NC}"
    echo "Success rate: $(( successful_experiments * 100 / total_experiments ))%"
    echo "Total time: ${hours}h ${minutes}m ${seconds}s"
    echo ""
    echo "Results directory: $RESULTS_DIR"
    echo "CSV logs directory: $CSV_DIR"
    echo "Log file: $LOG_FILE"
    echo "Summary file: $SUMMARY_FILE"
    echo ""
    
    # 列出所有结果文件
    echo -e "${BLUE}📁 Generated Files:${NC}"
    echo "PNG plots:"
    ls -la "$RESULTS_DIR"/*.png 2>/dev/null || echo "  No PNG files found"
    echo ""
    echo "CSV logs:"
    ls -la "$CSV_DIR"/*.csv 2>/dev/null || echo "  No CSV files found"
    echo ""
    
    # 生成简单的性能比较
    if [ -d "$CSV_DIR" ] && [ "$(ls -A $CSV_DIR)" ]; then
        echo -e "${BLUE}📊 Performance Summary:${NC}"
        echo "Mode,Scope,Batch_Size,Q,LR,Final_Loss,Avg_Loss"
        for csv in "$CSV_DIR"/*.csv; do
            if [ -f "$csv" ]; then
                # 提取文件名信息
                basename=$(basename "$csv" .csv)
                IFS='_' read -r mode scope bs_part q_part lr_part <<< "$basename"
                
                # 提取数值
                batch_size=$(echo "$bs_part" | sed 's/bs//')
                q=$(echo "$q_part" | sed 's/q//')
                lr=$(echo "$lr_part" | sed 's/lr//')
                
                # 从CSV中提取最终loss和平均loss
                if [ -f "$csv" ]; then
                    final_loss=$(tail -n +2 "$csv" | tail -n 1 | cut -d',' -f9)
                    avg_loss=$(tail -n +2 "$csv" | awk -F',' '{sum+=$9; count++} END {if(count>0) print sum/count; else print "N/A"}')
                    echo "$mode,$scope,$batch_size,$q,$lr,$final_loss,$avg_loss"
                fi
            fi
        done
    fi
}

# 主实验循环
echo -e "${BLUE}🔬 Starting experiments...${NC}"
echo ""

# 运行FO实验
echo -e "${GREEN}=== First-Order (FO) Experiments ===${NC}"
for scope in "${SCOPES[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for lr in "${LEARNING_RATES_FO[@]}"; do
            run_experiment "FO" "$scope" "$batch_size" "N/A" "$lr" "$EPOCHS"
        done
    done
done

echo ""

# 运行ZO实验
echo -e "${GREEN}=== Zeroth-Order (ZO) Experiments ===${NC}"
for scope in "${SCOPES[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for q in "${QUERY_BUDGETS[@]}"; do
            for lr in "${LEARNING_RATES_ZO[@]}"; do
                run_experiment "ZO" "$scope" "$batch_size" "$q" "$lr" "$EPOCHS"
            done
        done
    done
done

# 生成最终报告
echo ""
generate_report

echo -e "${GREEN}🎉 Sweep completed!${NC}"
echo "Check the results in the $RESULTS_DIR and $CSV_DIR directories."

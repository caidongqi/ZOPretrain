#!/bin/bash

# Parallel ZO vs FO Parameter Sweep Script
# 支持并行运行和GPU选择的参数扫描脚本

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 默认配置参数
MODES=("ZO" "FO")
SCOPES=("full")
BATCH_SIZES=(2)
QUERY_BUDGETS=(8)
LEARNING_RATES_ZO=(1e-6)
EPOCHS=10
LOG_INTERVAL=10

# 新增：优化器配置
OPTIMIZERS=("adam" "muon")  # 可用的优化器列表

# 并行配置
MAX_PARALLEL_JOBS=32 # 最大并行任务数
GPU_IDS="2,3"           # GPU ID列表，空表示自动检测

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        --modes)
            IFS=',' read -ra MODES <<< "$2"
            shift 2
            ;;
        --scopes)
            IFS=',' read -ra SCOPES <<< "$2"
            shift 2
            ;;
        --batch-sizes)
            IFS=',' read -ra BATCH_SIZES <<< "$2"
            shift 2
            ;;
        --query-budgets)
            IFS=',' read -ra QUERY_BUDGETS <<< "$2"
            shift 2
            ;;
        --learning-rates)
            IFS=',' read -ra LEARNING_RATES_ZO <<< "$2"
            shift 2
            ;;
        --optimizers)
            IFS=',' read -ra OPTIMIZERS <<< "$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --parallel N         最大并行任务数 (默认: 4)"
            echo "  --gpus '0,1,2'      指定GPU ID列表，支持逗号或空格分隔 (默认: 自动检测)"
            echo "  --modes 'FO,ZO'     优化方法 (默认: ZO)"
            echo "  --scopes 'reduced,full' 训练范围 (默认: reduced,full)"
            echo "  --batch-sizes '1,2,4' 批次大小 (默认: 1,2,4)"
            echo "  --query-budgets '1,2,4,8' Query budget (默认: 1,2,4,8)"
            echo "  --learning-rates '1e-4,1e-5' 学习率 (默认: 3e-4)"
            echo "  --optimizers 'adam,muon' 优化器 (默认: adam,muon)"
            echo "  --epochs N           训练轮数 (默认: 1)"
            echo "  --log-interval N     日志间隔 (默认: 10)"
            echo "  -h, --help           显示帮助信息"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 自动检测GPU
if [ -z "$GPU_IDS" ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [ $GPU_COUNT -gt 0 ]; then
            GPU_IDS=$(seq -s, 0 $((GPU_COUNT-1)))
            echo -e "${BLUE}🔍 Auto-detected $GPU_COUNT GPU(s): $GPU_IDS${NC}"
        else
            echo -e "${YELLOW}⚠️  No GPUs detected, using CPU${NC}"
            GPU_IDS="cpu"
        fi
    else
        echo -e "${YELLOW}⚠️  nvidia-smi not found, using CPU${NC}"
        GPU_IDS="cpu"
    fi
fi

# 创建结果目录
RESULTS_DIR="results"
CSV_DIR="csv_logs_mezo_epochs10_large"
CACHE_DIR="cache"
TEMP_DIR="temp"
mkdir -p "$RESULTS_DIR" "$CSV_DIR" "$CACHE_DIR" "$TEMP_DIR"

# 日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="parallel_sweep_${TIMESTAMP}.log"
SUMMARY_FILE="parallel_sweep_summary_${TIMESTAMP}.txt"
JOB_LOG_DIR="job_logs_${TIMESTAMP}"
mkdir -p "$JOB_LOG_DIR"

echo -e "${BLUE}🚀 Starting Parallel ZO vs FO Parameter Sweep${NC}"
echo -e "${BLUE}============================================${NC}"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "GPU IDs: $GPU_IDS"
echo "Results will be saved to: $RESULTS_DIR"
echo "CSV logs will be saved to: $CSV_DIR"
echo "Dataset cache: $CACHE_DIR"
echo "Log file: $LOG_FILE"
echo ""

# 生成所有实验配置
generate_experiments() {
    local experiments=()
    local exp_id=0
    
    for mode in "${MODES[@]}"; do
        for scope in "${SCOPES[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                if [ "$mode" = "ZO" ]; then
                    for q in "${QUERY_BUDGETS[@]}"; do
                        for lr in "${LEARNING_RATES_ZO[@]}"; do
                            for opt in "${OPTIMIZERS[@]}"; do
                                experiments+=("$exp_id:$mode:$scope:$batch_size:$q:$lr:$opt")
                                exp_id=$((exp_id + 1))
                            done
                        done
                    done
                else
                    # FO experiments
                    for lr in "${LEARNING_RATES_ZO[@]}"; do
                        for opt in "${OPTIMIZERS[@]}"; do
                            experiments+=("$exp_id:$mode:$scope:$batch_size:N/A:$lr:$opt")
                            exp_id=$((exp_id + 1))
                        done
                    done
                fi
            done
        done
    done
    
    printf '%s\n' "${experiments[@]}"
}

# 运行单个实验
run_single_experiment() {
    local exp_config="$1"
    local gpu_id="$2"
    
    IFS=':' read -r exp_id mode scope batch_size q lr opt <<< "$exp_config"
    
    local exp_name="${mode}_${scope}_bs${batch_size}_q${q}_lr${lr}_opt${opt}"
    local csv_file="${CSV_DIR}/${exp_name}.csv"
    local job_log="${JOB_LOG_DIR}/${exp_name}.log"
    
    echo -e "${YELLOW}📊 Starting experiment: $exp_name (GPU: $gpu_id)${NC}" | tee -a "$job_log"
    
    # 构建命令
    local cmd="python reproduce_zo_paper.py"
    cmd="$cmd --mode $mode"
    cmd="$cmd --scope $scope"
    cmd="$cmd --batch_size $batch_size"
    cmd="$cmd --learning_rate $lr"
    cmd="$cmd --optimizer $opt"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --csv_file $csv_file"
    cmd="$cmd --log_interval $LOG_INTERVAL"
    
    if [ "$mode" = "ZO" ] && [ "$q" != "N/A" ]; then
        cmd="$cmd --query_budget_q $q"
    fi
    
    # 设置GPU环境变量
    if [ "$gpu_id" != "cpu" ]; then
        export CUDA_VISIBLE_DEVICES="$gpu_id"
    else
        unset CUDA_VISIBLE_DEVICES
    fi
    
    echo "Command: $cmd" >> "$job_log"
    echo "GPU: $gpu_id" >> "$job_log"
    echo "Start time: $(date)" >> "$job_log"
    echo "----------------------------------------" >> "$job_log"
    
    # 运行实验
    if eval $cmd >> "$job_log" 2>&1; then
        echo -e "${GREEN}✅ Experiment $exp_name completed successfully${NC}" | tee -a "$job_log"
        echo "End time: $(date)" >> "$job_log"
        echo "SUCCESS" >> "$job_log"
        return 0
    else
        echo -e "${RED}❌ Experiment $exp_name failed${NC}" | tee -a "$job_log"
        echo "End time: $(date)" >> "$job_log"
        echo "FAILED" >> "$job_log"
        return 1
    fi
}

# 并行执行实验
run_parallel_experiments() {
    local experiments=($(generate_experiments))
    local total_experiments=${#experiments[@]}
    local completed=0
    local successful=0
    local failed=0
    
    echo -e "${BLUE}📋 Generated $total_experiments experiments${NC}"
    echo ""
    
    # 将GPU ID转换为数组（支持逗号和空格分隔）
    if [[ "$GPU_IDS" == *","* ]]; then
        IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    else
        IFS=' ' read -ra GPU_ARRAY <<< "$GPU_IDS"
    fi
    local gpu_count=${#GPU_ARRAY[@]}
    local gpu_index=0
    
    # 创建任务队列
    local job_queue=()
    local running_jobs=()
    
    # 初始化任务队列
    for exp in "${experiments[@]}"; do
        job_queue+=("$exp")
    done
    
    echo -e "${BLUE}🚀 Starting parallel execution...${NC}"
    echo ""
    
    # 主循环：管理并行任务
    while [ $completed -lt $total_experiments ]; do
        # 启动新任务（如果队列不为空且未达到最大并行数）
        while [ ${#running_jobs[@]} -lt $MAX_PARALLEL_JOBS ] && [ ${#job_queue[@]} -gt 0 ]; do
            local exp="${job_queue[0]}"
            job_queue=("${job_queue[@]:1}")  # 移除第一个元素
            
            local gpu_id="${GPU_ARRAY[$gpu_index]}"
            gpu_index=$(((gpu_index + 1) % gpu_count))
            
            # 在后台运行实验
            run_single_experiment "$exp" "$gpu_id" &
            local pid=$!
            running_jobs+=("$pid:$exp:$gpu_id")
            
            echo -e "${PURPLE}🔄 Started job $pid for experiment $exp on GPU $gpu_id${NC}"
        done
        
        # 检查完成的任务
        local new_running_jobs=()
        for job in "${running_jobs[@]}"; do
            IFS=':' read -r pid exp gpu_id <<< "$job"
            if kill -0 $pid 2>/dev/null; then
                # 任务仍在运行
                new_running_jobs+=("$job")
            else
                # 任务已完成
                wait $pid
                local exit_code=$?
                completed=$((completed + 1))
                
                if [ $exit_code -eq 0 ]; then
                    successful=$((successful + 1))
                else
                    failed=$((failed + 1))
                fi
                
                echo -e "${BLUE}📊 Progress: $completed/$total_experiments completed (Success: $successful, Failed: $failed)${NC}"
            fi
        done
        running_jobs=("${new_running_jobs[@]}")
        
        # 短暂等待
        sleep 1
    done
    
    # 等待所有剩余任务完成
    for job in "${running_jobs[@]}"; do
        IFS=':' read -r pid exp gpu_id <<< "$job"
        wait $pid
        local exit_code=$?
        completed=$((completed + 1))
        
        if [ $exit_code -eq 0 ]; then
            successful=$((successful + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    echo ""
    echo -e "${GREEN}🎉 All experiments completed!${NC}"
    echo "Total: $total_experiments, Success: $successful, Failed: $failed"
}

# 生成最终报告
generate_final_report() {
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo -e "${BLUE}📋 PARALLEL SWEEP SUMMARY REPORT${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
    echo "GPU IDs used: $GPU_IDS"
    echo "Total experiments: $total_experiments"
    echo -e "Successful: ${GREEN}$successful${NC}"
    echo -e "Failed: ${RED}$failed${NC}"
    echo "Success rate: $(( successful * 100 / total_experiments ))%"
    echo "Total time: ${hours}h ${minutes}m ${seconds}s"
    echo ""
    echo "Results directory: $RESULTS_DIR"
    echo "CSV logs directory: $CSV_DIR"
    echo "Job logs directory: $JOB_LOG_DIR"
    echo "Log file: $LOG_FILE"
    echo "Summary file: $SUMMARY_FILE"
    echo ""
    
    # 列出所有结果文件
    echo -e "${BLUE}📁 Generated Files:${NC}"
    echo "PNG plots:"
    ls -la "$RESULTS_DIR"/*.png 2>/dev/null | head -10 || echo "  No PNG files found"
    if [ $(ls -1 "$RESULTS_DIR"/*.png 2>/dev/null | wc -l) -gt 10 ]; then
        echo "  ... and $(($(ls -1 "$RESULTS_DIR"/*.png 2>/dev/null | wc -l) - 10)) more files"
    fi
    echo ""
    echo "CSV logs:"
    ls -la "$CSV_DIR"/*.csv 2>/dev/null | head -10 || echo "  No CSV files found"
    if [ $(ls -1 "$CSV_DIR"/*.csv 2>/dev/null | wc -l) -gt 10 ]; then
        echo "  ... and $(($(ls -1 "$CSV_DIR"/*.csv 2>/dev/null | wc -l) - 10)) more files"
    fi
    echo ""
}

# 主程序
main() {
    local start_time=$(date +%s)
    
    # 记录配置
    echo "Configuration:" >> "$LOG_FILE"
    echo "MODES: ${MODES[*]}" >> "$LOG_FILE"
    echo "SCOPES: ${SCOPES[*]}" >> "$LOG_FILE"
    echo "BATCH_SIZES: ${BATCH_SIZES[*]}" >> "$LOG_FILE"
    echo "QUERY_BUDGETS: ${QUERY_BUDGETS[*]}" >> "$LOG_FILE"
    echo "LEARNING_RATES_ZO: ${LEARNING_RATES_ZO[*]}" >> "$LOG_FILE"
    echo "OPTIMIZERS: ${OPTIMIZERS[*]}" >> "$LOG_FILE"
    echo "EPOCHS: $EPOCHS" >> "$LOG_FILE"
    echo "MAX_PARALLEL_JOBS: $MAX_PARALLEL_JOBS" >> "$LOG_FILE"
    echo "GPU_IDS: $GPU_IDS" >> "$LOG_FILE"
    echo "=========================================" >> "$LOG_FILE"
    
    # 运行并行实验
    run_parallel_experiments >> "$LOG_FILE" 2>&1
    
    # 生成报告
    generate_final_report >> "$LOG_FILE" 2>&1
    
    echo -e "${GREEN}🎉 Parallel sweep completed!${NC}"
    echo "Check the results in the $RESULTS_DIR and $CSV_DIR directories."
    echo "Detailed logs available in: $LOG_FILE"
}

# 运行主程序
main "$@"

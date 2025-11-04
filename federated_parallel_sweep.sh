#!/bin/bash

# Federated Learning (Flower) Parallel Sweep Script
# 参考 parallel_sweep.sh 的结构，针对 federated/server_flower.py 与 federated/client_flower.py

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 默认参数（可通过命令行覆盖）
MODES=("ZO" "FO")                   # 训练类型
SCOPES=("full")                     # 训练范围：full 或 reduced
BATCH_SIZES=(2)                      # 本地 batch_size
QUERY_BUDGETS=(1 8)                  # ZO 的 q（FO 忽略）
LEARNING_RATES=(1e-6)                # 学习率
OPTIMIZERS=("adam" "muon" "sgd")     # 优化器；其中 sgd 仅用于 ZO 且未启用 --zo-use-optimizer 时
LOCAL_EPOCHS=5                       # 客户端本地 epoch 数
LOCAL_STEPS=                         # 客户端本地步数上限（留空表示不限制）
SAMPLE_COUNT=20000                   # 每客户端样本量（用于构造缓存数据）
BLOCK_SIZE=128                       # token block size
ZO_USE_OPTIMIZER=                    # 若为 "true"，则 ZO 使用优化器更新；否则用手动 SGD
WEIGHT_DECAY=0.0
EPS=1e-8
BETAS="0.9,0.999"
MUON_CAUTIOUS=                       # 传入 --muon_cautious 开关
MUON_ORTHO_INIT=                     # 传入 --muon_orthogonal_init 开关
MUON_HIDDEN_SIZE=768

# 联邦相关
ROUNDS=3                             # Server 轮数
NUM_CLIENTS=1                   # 客户端数量
MIN_FIT_CLIENTS=1
MIN_AVAILABLE_CLIENTS=1
FRACTION_FIT=1.0
FRACTION_EVAL=0.0

# 并行与设备
MAX_PARALLEL_JOBS=8                  # 并行实验数
GPU_IDS="4,5"                        # 用于分配给客户端的 GPU 列表
DEVICE="cuda"                        # client device: auto|cpu|cuda

# 端口分配
SERVER_BASE_PORT=8190                # 每个实验 +exp_id 偏移，避免并发冲突
SERVER_HOST="127.0.0.1"             # Server 绑定主机

# 其他
LOG_INTERVAL=10
RESULTS_DIR="results"
JOB_LOG_DIR="fl_job_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR" "$JOB_LOG_DIR"

print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --parallel N                 最大并行实验数 (默认: $MAX_PARALLEL_JOBS)"
    echo "  --gpus '0,1,2'              指定客户端可用 GPU 列表 (默认: $GPU_IDS)"
    echo "  --modes 'ZO,FO'              模式列表 (默认: ${MODES[*]})"
    echo "  --scopes 'full,reduced'      训练范围列表 (默认: ${SCOPES[*]})"
    echo "  --batch-sizes '2,4'          本地 batch_size 列表 (默认: ${BATCH_SIZES[*]})"
    echo "  --query-budgets '1,8'        ZO 的 q 列表 (默认: ${QUERY_BUDGETS[*]})"
    echo "  --learning-rates '1e-6,3e-6' 学习率列表 (默认: ${LEARNING_RATES[*]})"
    echo "  --optimizers 'adam,muon,sgd' 优化器列表 (默认: ${OPTIMIZERS[*]})；sgd 仅用于 ZO 且不传 --zo-use-optimizer"
    echo "  --local-epochs N             本地 epochs (默认: $LOCAL_EPOCHS)"
    echo "  --local-steps N              本地步数上限，留空不限制 (默认: unset)"
    echo "  --sample-count N             每客户端样本量 (默认: $SAMPLE_COUNT)"
    echo "  --block-size N               block size (默认: $BLOCK_SIZE)"
    echo "  --zo-use-optimizer           ZO 使用优化器更新（默认: 关闭）"
    echo "  --weight-decay F             weight decay (默认: $WEIGHT_DECAY)"
    echo "  --eps F                      eps (默认: $EPS)"
    echo "  --betas '0.9,0.999'          betas (默认: $BETAS)"
    echo "  --muon-cautious              Muon Cautious 模式"
    echo "  --muon-ortho-init            Muon 正交初始化"
    echo "  --muon-hidden-size N         Muon 隐藏维度 (默认: $MUON_HIDDEN_SIZE)"
    echo "  --rounds N                   Server 轮数 (默认: $ROUNDS)"
    echo "  --num-clients N              客户端数量 (默认: $NUM_CLIENTS)"
    echo "  --min-fit-clients N          最小 fit 客户端 (默认: $MIN_FIT_CLIENTS)"
    echo "  --min-available-clients N    最小可用客户端 (默认: $MIN_AVAILABLE_CLIENTS)"
    echo "  --fraction-fit F             参与比例 (默认: $FRACTION_FIT)"
    echo "  --fraction-eval F            评估比例 (默认: $FRACTION_EVAL)"
    echo "  --device auto|cpu|cuda       客户端设备 (默认: $DEVICE)"
    echo "  --server-base-port N         Server 基础端口 (默认: $SERVER_BASE_PORT)"
    echo "  --server-host HOST           Server 主机 (默认: $SERVER_HOST)"
    echo "  --log-interval N             客户端日志间隔 (默认: $LOG_INTERVAL)"
    echo "  -h, --help                   显示帮助"
}

# 解析命令行
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            MAX_PARALLEL_JOBS="$2"; shift 2;;
        --gpus)
            GPU_IDS="$2"; shift 2;;
        --modes)
            IFS=',' read -ra MODES <<< "$2"; shift 2;;
        --scopes)
            IFS=',' read -ra SCOPES <<< "$2"; shift 2;;
        --batch-sizes)
            IFS=',' read -ra BATCH_SIZES <<< "$2"; shift 2;;
        --query-budgets)
            IFS=',' read -ra QUERY_BUDGETS <<< "$2"; shift 2;;
        --learning-rates)
            IFS=',' read -ra LEARNING_RATES <<< "$2"; shift 2;;
        --optimizers)
            IFS=',' read -ra OPTIMIZERS <<< "$2"; shift 2;;
        --local-epochs)
            LOCAL_EPOCHS="$2"; shift 2;;
        --local-steps)
            LOCAL_STEPS="$2"; shift 2;;
        --sample-count)
            SAMPLE_COUNT="$2"; shift 2;;
        --block-size)
            BLOCK_SIZE="$2"; shift 2;;
        --zo-use-optimizer)
            ZO_USE_OPTIMIZER="true"; shift 1;;
        --weight-decay)
            WEIGHT_DECAY="$2"; shift 2;;
        --eps)
            EPS="$2"; shift 2;;
        --betas)
            BETAS="$2"; shift 2;;
        --muon-cautious)
            MUON_CAUTIOUS="true"; shift 1;;
        --muon-ortho-init)
            MUON_ORTHO_INIT="true"; shift 1;;
        --muon-hidden-size)
            MUON_HIDDEN_SIZE="$2"; shift 2;;
        --rounds)
            ROUNDS="$2"; shift 2;;
        --num-clients)
            NUM_CLIENTS="$2"; shift 2;;
        --min-fit-clients)
            MIN_FIT_CLIENTS="$2"; shift 2;;
        --min-available-clients)
            MIN_AVAILABLE_CLIENTS="$2"; shift 2;;
        --fraction-fit)
            FRACTION_FIT="$2"; shift 2;;
        --fraction-eval)
            FRACTION_EVAL="$2"; shift 2;;
        --device)
            DEVICE="$2"; shift 2;;
        --server-base-port)
            SERVER_BASE_PORT="$2"; shift 2;;
        --server-host)
            SERVER_HOST="$2"; shift 2;;
        --log-interval)
            LOG_INTERVAL="$2"; shift 2;;
        -h|--help)
            print_help; exit 0;;
        *)
            echo "Unknown option: $1"; print_help; exit 1;;
    esac
done

# 将 GPU 列表解析成数组
if [[ "$GPU_IDS" == *","* ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
else
    IFS=' ' read -ra GPU_ARRAY <<< "$GPU_IDS"
fi
GPU_COUNT=${#GPU_ARRAY[@]}
if [ $GPU_COUNT -eq 0 ]; then
    echo -e "${YELLOW}⚠️ 未提供 GPU，将在 CPU 上运行客户端${NC}"
    GPU_ARRAY=("cpu")
    GPU_COUNT=1
fi

echo -e "${BLUE}🚀 Starting Federated Parallel Sweep${NC}"
echo "GPUs: ${GPU_ARRAY[*]}"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "Rounds: $ROUNDS, Num clients: $NUM_CLIENTS"
echo "Results dir: $RESULTS_DIR, Job logs: $JOB_LOG_DIR"

# 生成实验组合
generate_experiments() {
    local experiments=()
    local exp_id=0

    for mode in "${MODES[@]}"; do
        for scope in "${SCOPES[@]}"; do
            for bs in "${BATCH_SIZES[@]}"; do
                if [ "$mode" = "ZO" ]; then
                    for q in "${QUERY_BUDGETS[@]}"; do
                        for lr in "${LEARNING_RATES[@]}"; do
                            for opt in "${OPTIMIZERS[@]}"; do
                                experiments+=("$exp_id:$mode:$scope:$bs:$q:$lr:$opt")
                                exp_id=$((exp_id + 1))
                            done
                        done
                    done
                else
                    # FO：跳过不受支持的 sgd
                    for lr in "${LEARNING_RATES[@]}"; do
                        for opt in "${OPTIMIZERS[@]}"; do
                            if [ "$opt" = "sgd" ]; then
                                continue
                            fi
                            experiments+=("$exp_id:$mode:$scope:$bs:N/A:$lr:$opt")
                            exp_id=$((exp_id + 1))
                        done
                    done
                fi
            done
        done
    done

    printf '%s\n' "${experiments[@]}"
}

# 运行单个联邦实验：启动 server，再启动 N 个 client
run_single_fl_experiment() {
    local exp_config="$1"
    local log_prefix="$2"    # 用于 server 与 client 的日志文件前缀

    IFS=':' read -r exp_id mode scope bs q lr opt <<< "$exp_config"

    local port=$((SERVER_BASE_PORT + exp_id))
    local address="${SERVER_HOST}:${port}"
    local exp_name="FL_${mode}_${opt}_${scope}_n${NUM_CLIENTS}_q${q}_lr${lr}_e${LOCAL_EPOCHS}_bs${bs}"
    local server_log="${JOB_LOG_DIR}/${log_prefix}_${exp_name}_server.log"

    # 是否对当前实验启用 ZO 优化器：
    # 1) 全局显式指定 --zo-use-optimizer；或 2) ZO + muon 时自动启用
    local use_zo_optimizer=""
    if [ -n "$ZO_USE_OPTIMIZER" ]; then
        use_zo_optimizer="true"
    elif [ "$mode" = "ZO" ] && [ "$opt" = "muon" ]; then
        use_zo_optimizer="true"
    fi

    # 组合有效性检查：ZO + 启用优化器时不能选择 sgd
    if [ "$mode" = "ZO" ] && [ -n "$use_zo_optimizer" ] && [ "$opt" = "sgd" ]; then
        echo -e "${YELLOW}⏭️  Skip invalid combo: ZO with --zo-use-optimizer does not support 'sgd' optimizer${NC}" | tee -a "$server_log"
        return 0
    fi
    # FO 不支持 sgd（已在生成阶段过滤；此处再次保护）
    if [ "$mode" = "FO" ] && [ "$opt" = "sgd" ]; then
        echo -e "${YELLOW}⏭️  Skip invalid combo: FO does not support 'sgd' optimizer${NC}" | tee -a "$server_log"
        return 0
    fi

    echo -e "${YELLOW}📡 Launching server: $address for $exp_name${NC}" | tee -a "$server_log"

    # 启动 Server（后台）
    # 根据模式决定是否启用服务端 ZO 更新
    local server_cmd="python federated/server_flower.py --address $address --rounds $ROUNDS --min_fit_clients $MIN_FIT_CLIENTS --min_available_clients $MIN_AVAILABLE_CLIENTS --fraction_fit $FRACTION_FIT --fraction_evaluate $FRACTION_EVAL"
    if [ "$mode" = "ZO" ]; then
        server_cmd="$server_cmd --zo_server_side --zo_dir_count ${q#N/A} --zo_epsilon 1e-4 --zo_server_lr $lr"
    fi
    eval $server_cmd >> "$server_log" 2>&1 &
    local server_pid=$!

    # 给 server 一点时间启动监听
    sleep 2

    # 启动 Clients（后台）
    local client_pids=()
    local gpu_index=0
    for ((cid=0; cid<NUM_CLIENTS; cid++)); do
        local client_log="${JOB_LOG_DIR}/${log_prefix}_${exp_name}_client${cid}.log"
        local gpu_id="${GPU_ARRAY[$gpu_index]}"
        gpu_index=$(((gpu_index + 1) % GPU_COUNT))

        echo -e "${PURPLE}👤 Starting client ${cid} on GPU ${gpu_id}${NC}" | tee -a "$client_log"

        # 构建 client 命令
        local cmd="python federated/client_flower.py"
        cmd="$cmd --server $address"
        cmd="$cmd --client_id $cid"
        cmd="$cmd --num_clients $NUM_CLIENTS"
        cmd="$cmd --mode $mode"
        cmd="$cmd --scope $scope"
        if [ "$mode" = "ZO" ] && [ "$q" != "N/A" ]; then
            cmd="$cmd --q $q"
        fi
        cmd="$cmd --lr $lr"
        cmd="$cmd --local_epochs $LOCAL_EPOCHS"
        if [ -n "$LOCAL_STEPS" ]; then
            cmd="$cmd --local_steps $LOCAL_STEPS"
        fi
        cmd="$cmd --batch_size $bs"
        cmd="$cmd --block_size $BLOCK_SIZE"
        cmd="$cmd --cache_dir cache"
        cmd="$cmd --sample_count $SAMPLE_COUNT"
        # 仅在需要时传递 --optimizer：
        # - FO：必须传（adam/muon）
        # - ZO：当启用优化器路径时（全局或自动启用）且 opt 为 adam/muon 时传；
        #       若 opt 为 sgd 且未启用优化器，则不传此参数，客户端将走手动 SGD 路径
        if [ "$mode" = "FO" ]; then
            cmd="$cmd --optimizer $opt"
        else
            if [ -n "$use_zo_optimizer" ] && [ "$opt" != "sgd" ]; then
                cmd="$cmd --optimizer $opt"
            fi
        fi
        cmd="$cmd --weight_decay $WEIGHT_DECAY"
        cmd="$cmd --eps $EPS"
        # 处理 betas
        IFS=',' read -ra BETA_ARR <<< "$BETAS"
        if [ ${#BETA_ARR[@]} -eq 2 ]; then
            cmd="$cmd --betas ${BETA_ARR[0]} ${BETA_ARR[1]}"
        fi
        if [ -n "$MUON_CAUTIOUS" ]; then
            cmd="$cmd --muon_cautious"
        fi
        if [ -n "$MUON_ORTHO_INIT" ]; then
            cmd="$cmd --muon_orthogonal_init"
        fi
        cmd="$cmd --muon_hidden_size $MUON_HIDDEN_SIZE"
        if [ -n "$use_zo_optimizer" ]; then
            cmd="$cmd --zo_use_optimizer"
        fi
        cmd="$cmd --device $DEVICE"
        if [ "$DEVICE" = "cuda" ]; then
            cmd="$cmd --gpu $gpu_id"
        fi
        cmd="$cmd --log_interval $LOG_INTERVAL"

        if [ -z "$ZO_USE_OPTIMIZER" ] && [ -n "$use_zo_optimizer" ]; then
            echo "[auto] Enabled --zo_use_optimizer for ZO + muon" >> "$client_log"
        fi
        echo "Command: $cmd" >> "$client_log"
        eval $cmd >> "$client_log" 2>&1 &
        client_pids+=("$!")
    done

    # 等待所有客户端完成
    local exit_code=0
    for pid in "${client_pids[@]}"; do
        if ! wait $pid; then
            exit_code=1
        fi
    done

    # 等待 server（其会在 rounds 结束后退出）
    if ! wait $server_pid; then
        exit_code=1
    fi

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Experiment $exp_name completed successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Experiment $exp_name failed${NC}"
        return 1
    fi
}

# 并行调度多个实验
run_parallel() {
    local experiments=( $(generate_experiments) )
    local total=${#experiments[@]}
    local completed=0
    local successful=0
    local failed=0

    echo -e "${BLUE}📋 Generated $total experiments${NC}"

    local running_jobs=()

    while [ $completed -lt $total ]; do
        while [ ${#running_jobs[@]} -lt $MAX_PARALLEL_JOBS ] && [ ${#experiments[@]} -gt 0 ]; do
            local exp="${experiments[0]}"
            experiments=("${experiments[@]:1}")

            # 启动后台作业
            run_single_fl_experiment "$exp" "job$(date +%H%M%S)" &
            local pid=$!
            running_jobs+=("$pid:$exp")
            echo -e "${PURPLE}🔄 Started job $pid for exp $exp${NC}"
        done

        # 检查完成的任务
        local new_running=()
        for item in "${running_jobs[@]}"; do
            IFS=':' read -r pid exp <<< "$item"
            if kill -0 $pid 2>/dev/null; then
                new_running+=("$item")
            else
                wait $pid
                local code=$?
                completed=$((completed + 1))
                if [ $code -eq 0 ]; then
                    successful=$((successful + 1))
                else
                    failed=$((failed + 1))
                fi
                echo -e "${BLUE}📊 Progress: $completed/$total (Success: $successful, Failed: $failed)${NC}"
            fi
        done
        running_jobs=("${new_running[@]}")
        sleep 1
    done

    # 等待剩余任务
    for item in "${running_jobs[@]}"; do
        IFS=':' read -r pid exp <<< "$item"
        wait $pid
        local code=$?
        completed=$((completed + 1))
        if [ $code -eq 0 ]; then
            successful=$((successful + 1))
        else
            failed=$((failed + 1))
        fi
    done

    echo -e "${GREEN}🎉 All experiments completed! Total: $total, Success: $successful, Failed: $failed${NC}"
}

run_parallel



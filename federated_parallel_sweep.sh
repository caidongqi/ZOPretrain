#!/bin/bash

set -euo pipefail

WORKSPACE="/data/pc/ZOPretrain"
cd "$WORKSPACE"

PYTHON_BIN=${PYTHON_BIN:-python}

OPTIMIZERS=("adam")
ROUNDS=10000
LR="1e-3"
BATCH_SIZE=2
SCOPE="full"
SERVER_HOST="127.0.0.1"
BASE_PORT=8319
SERVER_DEVICE="auto"
CLIENT_DEVICE="auto"
CLIENT_Q=4
SAMPLE_COUNT=2048
BLOCK_SIZE=128
DIR_COUNT=1
EVAL_STEPS=1
INSTRUCT_CANDIDATE_POOL=64
LOG_INTERVAL=10
WEAK_COUNT=1
WAIT_SERVER_START=5
RESULT_ROOT="${WORKSPACE}/federated/parallel_runs"
CLIENT_GPU_LIST=(0 4 5)
# Auto GPU scheduling
AUTO_GPU=1
MIN_FREE_GB=5
MIN_FREE_GB_STRONG=12
GPU_POLL_INTERVAL=2
GPU_POLL_TIMEOUT=600
# Modes control: instruct, server_zo (comma-separated)
MODES="instruct"
# GPU launch lock and jitter
GPU_LOCK_GRACE=3
GPU_START_JITTER=0

print_help() {
  cat <<'EOF'
Usage: federated_parallel_sweep.sh [options]

Options:
  --rounds N            Number of federated rounds per experiment (default: 4)
  --optimizers LIST     Comma separated optimizers (default: sgd,adam,muon)
  --lr VALUE            Server-side learning rate (default: 1e-6)
  --batch-size N        Client batch size (default: 2)
  --base-port PORT      Starting TCP port for Flower server (default: 8300)
  --host HOST           Host/IP for Flower server binding (default: 127.0.0.1)
  --server-device DEV   Device flag for server process (default: auto)
  --client-device DEV   Default device flag for client processes (default: auto)
  --client-gpus LIST    Comma separated GPU ids (or cpu) assigned round-robin
  --sample-count N      Cached sample count per client (default: 2048)
  --block-size N        Token block size (default: 128)
  --scope VALUE         Trainable scope passed to clients (default: full)
  --dir-count N         Direction count for server-side ZO (default: 1)
  --eval-steps N        Batches per evaluation round (default: 1)
  --candidate-pool N    Instruct BP candidate pool size (default: 64)
  --log-interval N      Client CSV logging interval (default: 10)
  --weak-count N        Number of weak clients when mode=instruct (default: 8)
  --auto-gpu            Enable dynamic GPU scheduling by free memory (default: off)
  --min-free-gb N       Min free GB required to schedule a client (default: 5)
  --min-free-gb-strong N Min free GB required for strong client scheduling (default: 12)
  --gpu-poll-interval S Seconds between GPU free-mem checks (default: 2)
  --gpu-poll-timeout S  Max seconds to wait for a suitable GPU (default: 600)
  --modes LIST          Which modes to run: instruct,server_zo (default: both)
  --gpu-lock-grace S    Seconds to hold per-GPU lock after launch (default: 3)
  --gpu-start-jitter S  Random jitter [0,S] before/after launch (default: 0)
  --help                Show this message and exit

Environment overrides:
  PYTHON_BIN            Python executable to invoke (default: python)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rounds)
      ROUNDS="$2"
      shift 2
      ;;
        --optimizers)
      IFS=',' read -r -a OPTIMIZERS <<< "$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --host)
      SERVER_HOST="$2"
      shift 2
      ;;
    --server-device)
      SERVER_DEVICE="$2"
      shift 2
      ;;
    --client-device)
      CLIENT_DEVICE="$2"
      shift 2
      ;;
    --client-gpus)
      IFS=',' read -r -a CLIENT_GPU_LIST <<< "$2"
      shift 2
      ;;
        --sample-count)
      SAMPLE_COUNT="$2"
      shift 2
      ;;
        --block-size)
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --scope)
      SCOPE="$2"
      shift 2
      ;;
    --dir-count)
      DIR_COUNT="$2"
      shift 2
      ;;
    --eval-steps)
      EVAL_STEPS="$2"
      shift 2
      ;;
    --candidate-pool)
      INSTRUCT_CANDIDATE_POOL="$2"
      shift 2
      ;;
        --log-interval)
      LOG_INTERVAL="$2"
      shift 2
      ;;
    --weak-count)
      WEAK_COUNT="$2"
      shift 2
      ;;
    --auto-gpu)
      AUTO_GPU=1
      shift 1
      ;;
    --min-free-gb)
      MIN_FREE_GB="$2"
      shift 2
      ;;
    --min-free-gb-strong)
      MIN_FREE_GB_STRONG="$2"
      shift 2
      ;;
    --gpu-poll-interval)
      GPU_POLL_INTERVAL="$2"
      shift 2
      ;;
    --gpu-poll-timeout)
      GPU_POLL_TIMEOUT="$2"
      shift 2
      ;;
    --gpu-lock-grace)
      GPU_LOCK_GRACE="$2"
      shift 2
      ;;
    --gpu-start-jitter)
      GPU_START_JITTER="$2"
      shift 2
      ;;
    --help|-h)
      print_help
      exit 0
      ;;
    --modes)
      MODES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_help
      exit 1
      ;;
    esac
done

INSTRUCT_TOPK=$DIR_COUNT
mkdir -p "$RESULT_ROOT"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULT_ROOT}/${TIMESTAMP}"
mkdir -p "$RUN_DIR"
SUMMARY_FILE="${RUN_DIR}/loss_summary.csv"
echo "mode,optimizer,round,loss" > "$SUMMARY_FILE"

{
  echo "workspace=${WORKSPACE}"
  echo "timestamp=${TIMESTAMP}"
  echo "rounds=${ROUNDS}"
  echo "learning_rate=${LR}"
  echo "batch_size=${BATCH_SIZE}"
  echo "scope=${SCOPE}"
  echo "dir_count=${DIR_COUNT}"
  echo "eval_steps=${EVAL_STEPS}"
  echo "optimizers=$(IFS=,; echo "${OPTIMIZERS[*]}")"
  echo "server_device=${SERVER_DEVICE}"
  echo "client_device=${CLIENT_DEVICE}"
  echo "client_gpus=$(IFS=,; echo "${CLIENT_GPU_LIST[*]}")"
  echo "sample_count=${SAMPLE_COUNT}"
  echo "block_size=${BLOCK_SIZE}"
  echo "candidate_pool=${INSTRUCT_CANDIDATE_POOL}"
  echo "log_interval=${LOG_INTERVAL}"
  echo "weak_count=${WEAK_COUNT}"
  echo "auto_gpu=${AUTO_GPU}"
  echo "min_free_gb=${MIN_FREE_GB}"
  echo "min_free_gb_strong=${MIN_FREE_GB_STRONG}"
  echo "gpu_poll_interval=${GPU_POLL_INTERVAL}"
  echo "gpu_poll_timeout=${GPU_POLL_TIMEOUT}"
  echo "gpu_lock_grace=${GPU_LOCK_GRACE}"
  echo "gpu_start_jitter=${GPU_START_JITTER}"
} > "${RUN_DIR}/run_configuration.txt"

# --------------------------
# GPU scheduling helpers
# --------------------------
get_all_gpu_ids() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//'
}

get_gpu_free_mb_map() {
  # Output lines: "id free_mb"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{gsub(/ /,"",$0); print $1" "$2}'
}

pick_gpu_with_free_mb() {
  local min_free_mb="$1"
  local allowed_csv="$2" # comma-separated ids, may be empty meaning all
  local best_id=""
  local best_free=-1
  local allowed_pat="^("$(echo "$allowed_csv" | sed 's/,/|/g')")$"
  while read -r line; do
    local gid free
    gid=$(echo "$line" | awk '{print $1}')
    free=$(echo "$line" | awk '{print $2}')
    if [ -n "$allowed_csv" ]; then
      if ! echo "$gid" | grep -Eq "$allowed_pat"; then
        continue
      fi
    fi
    if [ "$free" -ge "$min_free_mb" ]; then
      if [ "$free" -gt "$best_free" ]; then
        best_free="$free"
        best_id="$gid"
      fi
    fi
  done < <(get_gpu_free_mb_map)
  echo "$best_id"
}

get_gpu_free_mb_single() {
  local gid="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gid" 2>/dev/null | head -n1 | tr -d '[:space:]'
}

run_experiment() {
  local mode="$1"
  local optimizer="$2"
  local port="$3"
  local index="$4"
  local total="$5"
  local gpu_list_str="$6"
  local address="${SERVER_HOST}:${port}"
  local exp_dir="${RUN_DIR}/${mode}_${optimizer}"
  mkdir -p "$exp_dir"

  local -a assigned_gpus=()
  if [ -n "$gpu_list_str" ]; then
    IFS=',' read -r -a assigned_gpus <<< "$gpu_list_str"
  fi

  local -a server_cmd=(
    "$PYTHON_BIN" "federated/server_flower.py"
    --address "$address"
    --rounds "$ROUNDS"
    --fraction_fit 1.0
    --fraction_evaluate 1.0
    --device "$SERVER_DEVICE"
    --server_zo_muon_hidden_size 768
  )

  if [ "$mode" = "instruct" ]; then
    server_cmd+=(
      --min_fit_clients $((1 + WEAK_COUNT))
      --min_available_clients $((1 + WEAK_COUNT))
      --instruct_enable
      --instruct_server_csv "${exp_dir}/server_round_metrics.csv"
      --instruct_candidate_pool "$INSTRUCT_CANDIDATE_POOL"
      --instruct_topk "$INSTRUCT_TOPK"
      --instruct_dir_count "$DIR_COUNT"
      --instruct_eval_steps "$EVAL_STEPS"
      --server_zo_lr "$LR"
      --server_zo_optimizer "$optimizer"
      --server_zo_dir_count "$DIR_COUNT"
      --server_zo_epsilon 1e-4
      --server_zo_betas 0.9 0.999
      --server_zo_eps 1e-8
      --server_zo_weight_decay 0.0
    )
  else
    server_cmd+=(
      --min_fit_clients 1
      --min_available_clients 1
      --server_zo_enable
      --server_zo_dir_count "$DIR_COUNT"
      --server_zo_lr "$LR"
      --server_zo_optimizer "$optimizer"
      --server_zo_epsilon 1e-4
      --server_zo_betas 0.9 0.999
      --server_zo_eps 1e-8
      --server_zo_weight_decay 0.0
    )
  fi

  local server_log="${exp_dir}/server.log"
  printf "\n=== [%d/%d] Running %s mode with %s optimizer on %s ===\n" "$index" "$total" "$mode" "$optimizer" "$address"
  printf "%s\n" "$(printf '%q ' "${server_cmd[@]}")" > "${exp_dir}/server_cmd.txt"

  "${server_cmd[@]}" > "$server_log" 2>&1 &
    local server_pid=$!

  sleep "$WAIT_SERVER_START"

  local client_total=1
  local roles=("weak")
  if [ "$mode" = "instruct" ]; then
    client_total=$((1 + WEAK_COUNT))
    roles=("strong")
    # append WEAK_COUNT times "weak"
    for ((wi=0; wi<WEAK_COUNT; wi++)); do
      roles+=("weak")
    done
  fi

  if [ "$AUTO_GPU" -eq 0 ]; then
    if [ "$client_total" -gt 0 ] && [ ${#assigned_gpus[@]} -lt "$client_total" ]; then
      echo "Error: not enough GPU assignments for ${mode}/${optimizer} (needed ${client_total}, got ${#assigned_gpus[@]})."
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
      return 1
    fi
  fi

    local client_pids=()
  local client_index=0

  for role in "${roles[@]}"; do
    local cid=$client_index
    local client_log="${exp_dir}/client_${cid}_${role}.log"
    local csv_path="${exp_dir}/client_${cid}_${role}.csv"
    local -a client_cmd=(
      "$PYTHON_BIN" "federated/client_flower.py"
      --server "$address"
      --client_id "$cid"
      --num_clients "$client_total"
      --mode "ZO"
      --scope "$SCOPE"
      --client_zo_q "$CLIENT_Q"
      --lr "$LR"
      --local_epochs 1
      --batch_size "$BATCH_SIZE"
      --block_size "$BLOCK_SIZE"
      --cache_dir "${WORKSPACE}/cache"
      --sample_count "$SAMPLE_COUNT"
      --log_interval "$LOG_INTERVAL"
      --csv_file "$csv_path"
      --role "$role"
    )

    local device_arg="$CLIENT_DEVICE"
    local -a device_flags=()

    local gpu_id=""
    if [ "$AUTO_GPU" -eq 1 ]; then
      # dynamic pick: choose a GPU with at least MIN_FREE_GB free, within CLIENT_GPU_LIST if provided
      local allowed_csv=""
      if [ ${#assigned_gpus[@]} -gt 0 ]; then
        allowed_csv=$(IFS=','; echo "${assigned_gpus[*]}")
      else
        # fall back to all GPUs
        allowed_csv="$(get_all_gpu_ids)"
      fi
      # role-specific threshold: strong uses higher bar
      local min_free_gb_role="$MIN_FREE_GB"
      if [ "$role" = "strong" ]; then
        min_free_gb_role="$MIN_FREE_GB_STRONG"
      fi
      local min_free_mb=$((min_free_gb_role * 1024))
      local waited=0
      local lock_fd=""
      local lock_path=""
      while true; do
        gpu_id="$(pick_gpu_with_free_mb "$min_free_mb" "$allowed_csv")"
        if [ -n "$gpu_id" ]; then
          # Acquire per-GPU lock to stagger starts
          lock_path="/tmp/fed_gpu_${gpu_id}.lock"
          # shellcheck disable=SC3020
          exec {lock_fd}> "$lock_path" || true
          if [ -n "$lock_fd" ]; then
            if flock -n "$lock_fd"; then
              # recheck free memory inside lock
              current_free="$(get_gpu_free_mb_single "$gpu_id")"
              if [ -z "$current_free" ]; then current_free=0; fi
              if [ "$current_free" -ge "$min_free_mb" ]; then
                break
              else
                # not enough now; release lock and continue polling
                flock -u "$lock_fd" || true
                exec {lock_fd}>&- || true
                lock_fd=""
                lock_path=""
              fi
            else
              # lock busy, continue polling
              exec {lock_fd}>&- || true
              lock_fd=""
            fi
          fi
        fi
        if [ "$waited" -ge "$GPU_POLL_TIMEOUT" ]; then
          echo "Timed out waiting for a GPU with >= ${MIN_FREE_GB}GB free for client ${client_index} (${mode}/${optimizer})."
          kill "$server_pid" 2>/dev/null || true
          wait "$server_pid" 2>/dev/null || true
          return 1
        fi
        sleep "$GPU_POLL_INTERVAL"
        waited=$((waited + GPU_POLL_INTERVAL))
      done
      # optional pre-launch jitter to further desynchronize
      if [ "$GPU_START_JITTER" != "0" ]; then
        # random in [0, GPU_START_JITTER]
        jitter=$(awk -v s="$GPU_START_JITTER" 'BEGIN{srand(); printf "%.3f", rand()*s}')
        sleep "$jitter"
      fi
      device_arg="cuda"
      device_flags+=(--gpu "$gpu_id")
    else
      if [ ${#assigned_gpus[@]} -gt "$client_index" ]; then
        gpu_id="${assigned_gpus[$client_index]}"
      fi
      if [ -n "$gpu_id" ]; then
        if [ "$gpu_id" = "cpu" ]; then
          device_arg="cpu"
        else
          device_arg="cuda"
          device_flags+=(--gpu "$gpu_id")
        fi
      fi
    fi

    client_cmd+=(--device "$device_arg")
    if [ ${#device_flags[@]} -gt 0 ]; then
      client_cmd+=("${device_flags[@]}")
    fi

    printf "%s\n" "$(printf '%q ' "${client_cmd[@]}")" > "${exp_dir}/client_${cid}_${role}_cmd.txt"

    "${client_cmd[@]}" > "$client_log" 2>&1 &
        client_pids+=("$!")
    # hold the lock for a grace period to allow memory allocation to complete
    if [ "$AUTO_GPU" -eq 1 ] && [ -n "$lock_fd" ]; then
      sleep "$GPU_LOCK_GRACE"
      flock -u "$lock_fd" || true
      exec {lock_fd}>&- || true
      lock_fd=""
    fi
    client_index=$((client_index + 1))
    done

  local client_status=0
    for pid in "${client_pids[@]}"; do
    if ! wait "$pid"; then
      client_status=1
        fi
    done

  if [ $client_status -ne 0 ]; then
    echo "Client failure detected; terminating server."
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    return 1
  fi

  if ! wait "$server_pid"; then
    echo "Server process exited with failure."
        return 1
    fi

  local loss_csv="${exp_dir}/round_losses.csv"
  local parse_output
  parse_output=$(
    "$PYTHON_BIN" - "$server_log" "$ROUNDS" "$loss_csv" "$mode" "$optimizer" \
      2> "${exp_dir}/loss_parse.stderr" <<'PY'
import sys, re, json, pathlib

log_path, rounds, csv_path, mode, opt = sys.argv[1:6]
rounds = int(rounds)
pattern = re.compile(r"loss\s*(?:=|:)\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
losses = []
with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if "evaluate" not in line.lower():
            continue
        match = pattern.search(line)
        if match:
            try:
                losses.append(float(match.group(1)))
            except ValueError:
                continue
if rounds > 0 and len(losses) > rounds:
    losses = losses[:rounds]
pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
with open(csv_path, "w", encoding="utf-8") as fh:
    fh.write("round,loss\n")
    for idx, value in enumerate(losses, start=1):
        fh.write(f"{idx},{value}\n")
with open(csv_path.replace(".csv", ".json"), "w", encoding="utf-8") as fh:
    json.dump({"mode": mode, "optimizer": opt, "losses": losses}, fh, indent=2)
print(len(losses))
PY
  )

  local parse_count
  parse_count=$(echo "$parse_output" | tr -d '[:space:]')
  if [[ ! "$parse_count" =~ ^[0-9]+$ ]]; then
    echo "Warning: unable to interpret evaluation loss count for ${mode}/${optimizer} (raw: ${parse_output})."
    parse_count=0
  fi

  if [ "$parse_count" -eq 0 ]; then
    echo "Warning: no evaluation losses were captured in server log for ${mode}/${optimizer}."
  elif [ "$parse_count" -ne "$ROUNDS" ]; then
    echo "Notice: parsed ${parse_count} evaluation losses (requested rounds: ${ROUNDS})."
  fi

  cat > "${exp_dir}/config.json" <<EOF
{
  "mode": "${mode}",
  "optimizer": "${optimizer}",
  "address": "${address}",
  "rounds": ${ROUNDS},
  "learning_rate": "${LR}",
  "batch_size": ${BATCH_SIZE},
  "scope": "${SCOPE}",
  "dir_count": ${DIR_COUNT},
  "eval_steps": ${EVAL_STEPS},
  "sample_count": ${SAMPLE_COUNT},
  "block_size": ${BLOCK_SIZE},
  "candidate_pool": ${INSTRUCT_CANDIDATE_POOL}
}
EOF

  printf "=== [%d/%d] Finished %s/%s ===\n" "$index" "$total" "$mode" "$optimizer"
}

declare -a EXPERIMENTS=()
declare -a EXPERIMENT_CLIENT_COUNTS=()
declare -a EXPERIMENT_PORTS=()

IFS=',' read -r -a MODES_ARR <<< "$MODES"
want_instruct=0
want_server_zo=0
for m in "${MODES_ARR[@]}"; do
  if [ "$m" = "instruct" ]; then want_instruct=1; fi
  if [ "$m" = "server_zo" ]; then want_server_zo=1; fi
done

for optimizer in "${OPTIMIZERS[@]}"; do
  if [ "$want_instruct" -eq 1 ]; then
    local_idx=${#EXPERIMENTS[@]}
    EXPERIMENTS+=("instruct:${optimizer}")
    EXPERIMENT_CLIENT_COUNTS+=($((1 + WEAK_COUNT)))
    EXPERIMENT_PORTS+=($((BASE_PORT + local_idx)))
  fi
  if [ "$want_server_zo" -eq 1 ]; then
    local_idx=${#EXPERIMENTS[@]}
    EXPERIMENTS+=("server_zo:${optimizer}")
    EXPERIMENT_CLIENT_COUNTS+=(1)
    EXPERIMENT_PORTS+=($((BASE_PORT + local_idx)))
  fi
done

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
echo "Total experiments: ${TOTAL_EXPERIMENTS}"

if [ "$TOTAL_EXPERIMENTS" -eq 0 ]; then
  echo "No experiments scheduled. Exiting."
  exit 0
fi

max_clients_per_experiment=0
for count in "${EXPERIMENT_CLIENT_COUNTS[@]}"; do
  if [ "$count" -gt "$max_clients_per_experiment" ]; then
    max_clients_per_experiment="$count"
  fi
done

if [ "$AUTO_GPU" -eq 0 ]; then
  if [ ${#CLIENT_GPU_LIST[@]} -lt "$max_clients_per_experiment" ]; then
    echo "Error: 每个实验最多需要 ${max_clients_per_experiment} 张卡，但当前 GPU 列表只有 ${#CLIENT_GPU_LIST[@]} 项。可启用 --auto-gpu 或扩充 --client-gpus。"
    exit 1
  fi
fi

declare -a AVAILABLE_GPUS=("${CLIENT_GPU_LIST[@]}")
declare -a RUNNING_PIDS=()
declare -a RUNNING_LABELS=()
declare -a RUNNING_GPUS=()

failures=0
next_experiment_idx=0

while [ "$next_experiment_idx" -lt "$TOTAL_EXPERIMENTS" ] || [ ${#RUNNING_PIDS[@]} -gt 0 ]; do
  scheduled=false

  while [ "$next_experiment_idx" -lt "$TOTAL_EXPERIMENTS" ]; do
    clients_needed=${EXPERIMENT_CLIENT_COUNTS[$next_experiment_idx]}
    if [ "$AUTO_GPU" -eq 0 ]; then
      if [ ${#AVAILABLE_GPUS[@]} -lt "$clients_needed" ]; then
        break
      fi
    fi

    declare -a assigned=()
    if [ "$AUTO_GPU" -eq 0 ]; then
      for ((i=0; i<clients_needed; i++)); do
        assigned+=("${AVAILABLE_GPUS[0]}")
        if [ ${#AVAILABLE_GPUS[@]} -gt 1 ]; then
          AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
        else
          AVAILABLE_GPUS=()
        fi
      done
    else
      # In auto mode, pass allowed GPU set (if any) through assigned to limit selection;
      # if CLIENT_GPU_LIST is empty, assigned stays empty to allow all GPUs.
      if [ ${#AVAILABLE_GPUS[@]} -gt 0 ]; then
        assigned=("${AVAILABLE_GPUS[@]}")
      fi
    fi

    label="${EXPERIMENTS[$next_experiment_idx]}"
    port="${EXPERIMENT_PORTS[$next_experiment_idx]}"
    index_display=$((next_experiment_idx + 1))
    gpu_str=$(IFS=','; echo "${assigned[*]}")
    printf "Assigning GPUs [%s] to %s (port %s)\n" "$gpu_str" "$label" "$port"

    IFS=: read -r mode optimizer <<< "$label"
    run_experiment "$mode" "$optimizer" "$port" "$index_display" "$TOTAL_EXPERIMENTS" "$gpu_str" &
    pid=$!

    RUNNING_PIDS+=("$pid")
    RUNNING_LABELS+=("$label")
    RUNNING_GPUS+=("$gpu_str")

    next_experiment_idx=$((next_experiment_idx + 1))
    scheduled=true
  done

  if [ ${#RUNNING_PIDS[@]} -gt 0 ]; then
    pid=${RUNNING_PIDS[0]}
    label=${RUNNING_LABELS[0]}
    gpu_str=${RUNNING_GPUS[0]}

    if ! wait "$pid"; then
      echo "Experiment ${label} failed."
      failures=$((failures + 1))
    fi

    if [ "$AUTO_GPU" -eq 0 ]; then
      if [ -n "$gpu_str" ]; then
        IFS=',' read -r -a released <<< "$gpu_str"
        AVAILABLE_GPUS+=("${released[@]}")
      fi
    fi

    RUNNING_PIDS=("${RUNNING_PIDS[@]:1}")
    RUNNING_LABELS=("${RUNNING_LABELS[@]:1}")
    RUNNING_GPUS=("${RUNNING_GPUS[@]:1}")
  elif [ "$scheduled" = false ] && [ "$next_experiment_idx" -lt "$TOTAL_EXPERIMENTS" ]; then
    if [ "$AUTO_GPU" -eq 0 ]; then
      echo "Error: 当前可用 GPU 数不足以启动实验 ${EXPERIMENTS[$next_experiment_idx]}（需要 ${EXPERIMENT_CLIENT_COUNTS[$next_experiment_idx]} 张，现余 ${#AVAILABLE_GPUS[@]} 张）。"
      failures=$((failures + 1))
      break
    else
      # In auto mode, if scheduling failed unexpectedly, break to avoid busy loop
      echo "Notice: 调度未成功，进入等待。"
      sleep 2
    fi
  else
    break
  fi
done

if [ "$failures" -gt 0 ]; then
  echo "${failures} experiment(s) failed."
  exit 1
fi

for idx in "${!EXPERIMENTS[@]}"; do
  IFS=: read -r mode optimizer <<< "${EXPERIMENTS[$idx]}"
  loss_csv="${RUN_DIR}/${mode}_${optimizer}/round_losses.csv"
  if [ -s "$loss_csv" ]; then
    tail -n +2 "$loss_csv" | while IFS=, read -r round loss; do
      if [ -n "$round" ] && [ -n "$loss" ]; then
        echo "${mode},${optimizer},${round},${loss}" >> "$SUMMARY_FILE"
      fi
    done
  fi
done

echo
echo "All experiments finished."
echo "Summary (mode, optimizer, round, loss):"
if [ -s "$SUMMARY_FILE" ]; then
  if command -v column >/dev/null 2>&1; then
    column -t -s, "$SUMMARY_FILE"
  else
    cat "$SUMMARY_FILE"
  fi
else
  echo "No evaluation loss entries were captured. Inspect logs under ${RUN_DIR}."
fi

echo
echo "Detailed logs and configs are stored in: ${RUN_DIR}"


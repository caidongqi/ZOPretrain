# ZOPretrain Quickstart

> Make sure you can reach the target host through the internal network.  
> Join the TailScale network first: https://login.tailscale.com/admin/invite/9EymnPL1yyZEY5msYcby11

## Prerequisites

1. Install TailScale and verify connectivity to the remote host.
2. Clone the repository:
   ```bash
   git clone https://github.com/caidongqi/ZOPretrain.git
   cd ZOPretrain
   ```
3. Create and activate a Conda environment (example name `flwr`):
   ```bash
   conda create -n flwr python=3.10
   conda activate flwr
   python -m pip install -U pip setuptools wheel
   pip install -r requirements-mac-cpu.txt
   ```

## Run the Server (example port 8319)

```bash
python /data/pc/ZOPretrain/federated/server_flower.py \
  --address 0.0.0.0:8319 \
  --rounds 10000 \
  --fraction_fit 1.0 \
  --fraction_evaluate 1.0 \
  --device auto \
  --min_fit_clients 2 \
  --min_available_clients 2 \
  --instruct_enable \
  --instruct_server_csv /data/pc/ZOPretrain/federated/parallel_runs/server_round_metrics.csv \
  --instruct_candidate_pool 64 \
  --instruct_topk 1 \
  --instruct_dir_count 1 \
  --instruct_eval_steps 1 \
  --server_zo_lr 1e-3 \
  --server_zo_optimizer adam \
  --server_zo_dir_count 1 \
  --server_zo_epsilon 1e-4 \
  --server_zo_betas 0.9 0.999 \
  --server_zo_eps 1e-8 \
  --server_zo_weight_decay 0.0
```

Start the strong client (`client_id=0`) on the same server:

```bash
python /data/pc/ZOPretrain/federated/client_flower.py \
  --server 127.0.0.1:8319 \
  --client_id 0 \
  --num_clients 2 \
  --mode ZO \
  --scope full \
  --client_zo_q 4 \
  --lr 1e-3 \
  --local_epochs 1 \
  --batch_size 2 \
  --block_size 128 \
  --cache_dir /data/pc/ZOPretrain/cache \
  --sample_count 2048 \
  --log_interval 10 \
  --csv_file /data/pc/ZOPretrain/federated/parallel_runs/client_0_strong.csv \
  --role strong \
  --device auto
```

## Run the Weak Client on macOS

1. Forward the server port via SSH:
   ```bash
   ssh Phoenix22 -N -L 8319:127.0.0.1:8319
   ```
2. Launch the weak client (`client_id=1`) locally:
   ```bash
   cd ZOPretrain
   python federated/client_flower.py \
     --server 127.0.0.1:8319 \
     --client_id 1 \
     --num_clients 2 \
     --mode ZO \
     --scope full \
     --lr 1e-3 \
     --local_epochs 1 \
     --batch_size 2 \
     --block_size 128 \
     --cache_dir cache \
     --sample_count 2048 \
     --log_interval 10 \
     --csv_file federated/parallel_runs/client_1_weak.csv \
     --role weak \
     --device auto
   ```

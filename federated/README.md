## 联邦学习（Flower）集成

本目录提供使用 Flower (FL) 的最小可运行示例，复用 `reproduce_zo_paper.py` 的模型与 ZO/FO 训练逻辑，实现客户端只同步“可训练子集参数”的 FedAvg/ZO。

### 安装

```bash
pip install -r federated/requirements_federated.txt
```

### 参数一览（明确 Server / Client 归属）

- Server 侧（`federated/server_flower.py`）
  - `--server_zo_enable`: 启用服务端 ZO。启用后客户端不本地更新，只回报方向导数标量。
  - `--server_zo_dir_count`: 每轮方向数（服务端生成相同随机种子并下发）。
  - `--server_zo_epsilon`: 客户端在 f(x±εu) 评估时使用的 ε（服务端下发）。
  - `--server_zo_lr`: 服务端应用 ZO 梯度更新时使用的学习率。
  - `--server_zo_optimizer`: `sgd|adam|muon`，服务端应用更新所用优化器。
  - `--server_zo_weight_decay`, `--server_zo_eps`, `--server_zo_betas`: Adam/Muon 的正则与数值稳定参数。
  - `--server_zo_muon_cautious`, `--server_zo_muon_orthogonal_init`, `--server_zo_muon_hidden_size`: Muon 专属选项。
  - 其余 Flower 通用：`--address --rounds --fraction_fit --min_fit_clients --min_available_clients --device` 等。

- Client 侧（`federated/client_flower.py`）
  - `--mode`: `FO|ZO`。FO 为一阶优化；ZO 为零阶优化。
  - `--client_zo_q`（别名 `--q`）: 仅在“客户端本地 ZO”时生效；表示每步采样的方向数。若开启 `--server_zo_enable`（服务端 ZO），该参数被忽略。
  - `--lr`: 客户端本地更新学习率（仅 FO 或 ZO+本地优化器/手动SGD 生效）。
  - `--optimizer --zo_use_optimizer --weight_decay --eps --betas --muon_*`: 客户端本地优化器配置。
  - 数据与训练控制：`--scope --batch_size --block_size --cache_dir --sample_count --local_epochs --local_steps --device`。

提示：服务端 ZO 模式下，方向数由 `--server_zo_dir_count` 决定；客户端的 `--client_zo_q` 不参与更新。

### 常用场景与示例命令

1) 服务端 ZO（客户端只评估并回报方向导数）

Server：
```bash
python federated/server_flower.py \
  --address 0.0.0.0:8089 --rounds 5 --min_fit_clients 1 --min_available_clients 1 \
  --fraction_fit 1.0 --fraction_evaluate 0.0 --device cuda \
  --server_zo_enable \
  --server_zo_dir_count 3 \
  --server_zo_epsilon 1e-4 \
  --server_zo_lr 1e-6 \
  --server_zo_optimizer adam \
  --server_zo_betas 0.9 0.999 --server_zo_eps 1e-8 --server_zo_weight_decay 0.0
```

Client：
```bash
python federated/client_flower.py \
  --server 127.0.0.1:8089 --client_id 0 --num_clients 1 \
  --mode ZO --scope reduced --client_zo_q 3 \
  --lr 1e-6 --local_epochs 1 --local_steps 1 \
  --batch_size 2 --block_size 64 --cache_dir cache --sample_count 16 --device cuda
```
说明：此模式下 `--client_zo_q` 被忽略（以服务端下发的 `--server_zo_dir_count` 为准）。

2) 客户端本地 ZO（不启用服务端 ZO，客户端本地采样方向并更新）

Server：
```bash
python federated/server_flower.py \
  --address 0.0.0.0:8089 --rounds 5 --min_fit_clients 1 --min_available_clients 1 \
  --fraction_fit 1.0 --fraction_evaluate 0.0 --device cuda
```

Client（手动 SGD 更新）：
```bash
python federated/client_flower.py \
  --server 127.0.0.1:8089 --client_id 0 --num_clients 1 \
  --mode ZO --scope reduced --client_zo_q 3 --lr 1e-6 \
  --local_epochs 1 --local_steps 50 --batch_size 8 --block_size 128 --device cuda
```

Client（用 Adam/Muon 更新）：
```bash
python federated/client_flower.py \
  --server 127.0.0.1:8089 --client_id 0 --num_clients 1 \
  --mode ZO --scope reduced --client_zo_q 3 --lr 1e-6 \
  --zo_use_optimizer --optimizer adam --betas 0.9 0.999 --eps 1e-8 --weight_decay 0.0 \
  --local_epochs 1 --local_steps 50 --batch_size 8 --block_size 128 --device cuda
```

### 设计说明

- 服务端 ZO：服务端产生随机方向种子并下发，客户端仅做 f(x±εu) 评估，返回标量导数列表；服务端重建方向并用选定优化器（SGD/Adam/Muon）更新全局参数。
- 客户端本地 ZO：客户端以 `--client_zo_q` 采样方向估计梯度并本地更新（手动 SGD 或指定优化器）；常规 FedAvg 聚合参数。
- 客户端与服务器仅交换“可训练参数子集”的权重；默认 `scope=reduced` 以降通信成本。

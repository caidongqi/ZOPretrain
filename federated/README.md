## 联邦学习（Flower）集成

本目录提供使用 Flower (FL) 的最小可运行示例，复用 `reproduce_zo_paper.py` 的模型与 ZO/FO 训练逻辑，实现客户端只同步“可训练子集参数”的 FedAvg。

### 安装

```bash
pip install -r federated/requirements_federated.txt
```

### 启动服务器

```bash
python federated/server_flower.py --address 0.0.0.0:8089 --rounds 3 \
  --min_fit_clients 1 --min_available_clients 1 --fraction_fit 1.0
```

### 启动客户端（示例：2 个客户端）

分别在两个终端中运行：

```bash
python federated/client_flower.py --server 127.0.0.1:8089 \
  --client_id 0 --num_clients 1 --mode ZO --scope reduced \
  --q 1 --lr 1e-5 --local_epochs 1 --local_steps 50 \
  --batch_size 8 --block_size 128 --cache_dir cache --sample_count 20000

python federated/client_flower.py --server 127.0.0.1:8080 \
  --client_id 1 --num_clients 2 --mode ZO --scope reduced \
  --q 1 --lr 1e-5 --local_epochs 1 --local_steps 50 \
  --batch_size 8 --block_size 128 --cache_dir cache --sample_count 20000
```

说明：
- **mode**: `ZO` 或 `FO`；
- **scope**: `reduced`（仅最后一层 MLP+LN+lm_head）或 `full`；
- **q**: ZO 方向数；`FO` 时忽略；
- **local_steps**: 每轮最多本地步数（可选，限制训练成本）；
- **zo_use_optimizer**: 使 ZO 使用优化器（如 Adam/Muon）而非手动 SGD。

### 设计说明

- 客户端与服务器仅交换“可训练参数子集”的权重向量，默认 `scope=reduced`，通信成本更低；
- 客户端数据通过 `federated/data_utils.py` 基于缓存的 IID 切片构造；
- 训练核心复用 `reproduce_zo_paper.py` 的 `create_model`、`get_trainable_parameters`、`zo_gradient_estimator`；
- 评估阶段仅取少量步（默认 20）快速估算损失，避免开销。


bash federated_parallel_sweep.sh \
  --modes 'ZO' \
  --optimizers 'adam,muon,sgd' \
  --learning-rates '1e-6,3e-6' \
  --query-budgets '1,2' \
  --num-clients 1 --rounds 5 \
  --local-epochs 5 --batch-sizes '2' \
  --parallel 5 --log-interval 10
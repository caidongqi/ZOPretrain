## federated_parallel_sweep.sh 使用说明

本文档说明如何使用 `federated_parallel_sweep.sh` 对比 Instruct 与纯服务端零阶（server-side ZO）两套联邦训练流程，并对关键超参数做批量实验。

---

### 1. 脚本位置与依赖

- 脚本路径：`/data/pc/ZOPretrain/federated_parallel_sweep.sh`
- 运行前请确认已安装必要依赖：
  ```bash
  pip install -r federated/requirements_federated.txt
  pip install torch
  ```
- 推荐提前准备好模型与数据缓存（默认使用 `cache/` 目录）。

---

### 2. 核心功能

每次调用脚本会按顺序运行以下实验，并将结果保存在独立时间戳目录下：

1. **Instruct 模式**：服务器启用两阶段 BP/ZO，启动 strong/weak 两个客户端。
2. **仅 server-side ZO**：服务器完全负责梯度更新，客户端只做有限差分评估（单个客户端）。

两种模式均会遍历设定的服务端优化器列表（默认 `sgd`、`adam`、`muon`），最终在 `loss_summary.csv` 中给出每轮 loss。

目录结构示意：

```
federated/parallel_runs/<timestamp>/
  ├─ run_configuration.txt     # 本次 sweep 的全局配置快照
  ├─ loss_summary.csv          # 汇总：mode, optimizer, round, loss
  ├─ instruct_<opt>/           # 按模式+优化器划分的子目录
  │    ├─ server.log / client_*.log
  │    ├─ round_losses.csv/.json
  │    └─ 命令行快照、配置快照
  └─ server_zo_<opt>/          # 同上
```

---

### 3. 常用命令

#### 3.1 默认运行（全流程对比）

```
bash federated_parallel_sweep.sh
```

#### 3.2 指定 GPU / CPU、减小样本量加速调试

```
bash federated_parallel_sweep.sh \
  --client-gpus 0,1 \
  --rounds 2 \
  --sample-count 512 \
  --client-device cuda \
  --server-device cuda
```

- `--client-gpus cpu` 可强制客户端跑在 CPU；
- `--rounds` 控制联邦轮次；
- `--sample-count` 越小越快，适合验证流程。

#### 3.3 自定义优化器集合与学习率

```
bash federated_parallel_sweep.sh \
  --optimizers sgd,muon \
  --lr 5e-7
```

---

### 4. 参数解释

可用 `--help` 查看完整参数，核心项如下：

| 参数 | 默认值 | 说明 |
| ---- | ---- | ---- |
| `--optimizers` | `sgd,adam,muon` | 服务端使用的优化器列表，每个优化器都会跑 Instruct + server_zo 两组实验 |
| `--rounds` | `4` | Flower 训练轮次，自动触发评估并记录 loss |
| `--lr` | `1e-6` | 服务端应用 ZO 梯度的学习率 |
| `--dir-count` | `1` | 服务端 ZO 的方向数，Instruct 模式也会同步使用 |
| `--candidate-pool` | `64` | Instruct strong 客户端 BP 阶段抽样的候选方向种子数 |
| `--sample-count` | `2048` | 每客户端采样数据量，用于构造缓存 |
| `--batch-size` | `2` | 客户端前向 batch size |
| `--client-gpus` | 空 | 客户端轮流绑定的 GPU ID 列表；可填 `cpu` |
| `--client-device` | `auto` | Flower 客户端的 `--device` 传参 |
| `--server-device` | `auto` | Flower 服务器的 `--device` 传参 |
| `--base-port` | `8300` | Flower server 起始端口，不同实验递增 |

---

### 5. Sweep 示例

脚本一次只接受一组参数，若要做多组 sweep，可在外层写循环：

```
for dir in 1 4; do
  for cand in 64 128; do
    bash federated_parallel_sweep.sh \
      --dir-count "$dir" \
      --candidate-pool "$cand" \
      --rounds 4
  done
done
```

也可以叠加学习率、batch size 等自定义组合，最终比较不同目录下的 `loss_summary.csv`。

---

### 6. 输出解读

- `loss_summary.csv`：按 `mode, optimizer, round, loss` 记录所有实验；建议用表格或 `column -t` 查看。
- `round_losses.csv/.json`：每组实验的详细 loss 序列。
- `server.log` & `client_*.log`：原始日志，包含通信量、评估详情等。

若解析失败（例如日志缺失 loss 字段），脚本会在终端给出提示，可手动检查对应日志文件。

---

### 7. 常见问题

1. **提示缺少 torch / flwr**：请确认依赖安装完整。
2. **端口冲突**：调整 `--base-port`，或避免同时运行多个 sweep。
3. **日志中无 loss**：确保 Flower 服务器成功走到评估阶段（脚本默认设置 `fraction_evaluate=1.0`，但若客户端异常退出，评估可能跳过）。
4. **GPU 不够用**：可将 `--client-device cpu`，或改用 `--client-gpus cpu`。Instruct 模式会创建两个客户端，需要至少两个进程同时运行。

---

如需按更复杂的组合扫描（例如对比多个 batch size、不同模式下的方向数设置等），可在上层调用脚本前先拼装参数列表；所有实验结果均按照时间戳隔离，便于后续比对与复现实验。祝调参顺利 🚀


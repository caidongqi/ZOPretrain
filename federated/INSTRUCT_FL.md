## Instruct 指导 ZO 训练（强/弱客户端）使用说明

### 背景
- Strong 客户端：计算一次 BP 梯度，生成与该梯度具有目标余弦相似度（默认 0.9）的多个方向，并回传给 Server。
- Server：汇总 strong 的“指令方向”，在下一轮将这些方向分发给 weak 客户端。
- Weak 客户端：仅沿接收到的方向做前向有限差分评估，回传每个方向的标量导数；Server 重建全量梯度并更新全局参数。

### 依赖安装
- 推荐先安装联邦依赖：
  - `pip install -r federated/requirements_federated.txt`
- 若使用 Muon：
  - `optim_muon.py` 已包含所需实现（无需额外安装）。

### 快速开始（单机）
1. 运行调试脚本（会同时启动 server/strong/weak）：
   - `bash instruct_debug.sh`
2. 观察日志：
   - `instruct_debug_logs/server.log`：每轮打印
     - `[Instruct][BP-Select] selected=... dirs=... avg_c->s=...B avg_s->c=...B`
     - `[Instruct][ZO-Eval] dirs=... avg_c->s=...B avg_s->c=...B`
   - strong/weak 客户端日志：
     - 回传 `comm_in_bytes/comm_out_bytes`
     - strong 返回 `instruct_dir_blob_json`（方向序列化后的控制负载规模）
     - weak 返回 `zo_dir_g_json`（每个方向的标量导数）

### 关键参数说明
- Server 侧（`federated/server_flower.py`）：
  - `--instruct_enable`：启用两阶段 Instruct（奇数轮 BP，偶数轮 ZO）。
  - `--instruct_candidate_pool`：BP 阶段 strong 评估的候选方向种子数量。
  - `--instruct_topk`：strong 回传给 server 的 top-k 种子（仅用于回退）。
  - `--instruct_dir_count`：最终下发给 weak 的方向数量（也是 weak 的评估个数）。
  - `--instruct_eval_steps`：weak 每轮评估使用的 batch 步数，建议从 1 起。
  - `--server_zo_epsilon`：有限差分步长 ε。
  - `--server_zo_lr` / `--server_zo_optimizer`：server 侧应用 ZO 梯度的优化器与学习率。

- Client 侧（`federated/client_flower.py`）：
  - `--role strong|weak`：指定客户端角色。
  - `--mode ZO --scope reduced`：建议 reduced 以减小通信与计算。
  - `--sample_count --batch_size --block_size --local_steps`：可增减以调试速度/稳定性。

### 工作流程
- 轮次 1（bp_select）：server 下发候选种子池 → strong 计算一次 BP 梯度 → 生成若干“指令方向”（与 BP 梯度余弦相似度接近目标）→ 回传给 server。
- 轮次 2（zo_eval）：server 将“指令方向”下发给 weak → weak 做 f(x±εd) 前向评估 → 回传每个方向的标量导数 → server 重建梯度并更新。
- 轮次 3/4：重复上述过程。

### 与生成方向相关的实现
- 精确余弦控制（推荐）：`bp_instruct.generate_instruct_directions`
  - 构造 d = c·g + sqrt(1−c²)·||g||·n（n 与 g 正交），cos(d,g) ≈ c。
- 稀疏/能量阈值法：`bp_instruct.generate_instruct_directions_with_R`
  - 通过能量覆盖率选择坐标，确保 cos(d,g) ≥ 目标下限，支持 `max_rank/seed/tune_to_target` 等。

### 常见问题
- 看不到方向 blob：检查 strong 客户端日志是否出现 `instruct_dir_blob_json`；若为空则回退到种子方式（服务端会下发 `zo_dir_seeds_json`）。
- 通信统计为 0：确保查看的是同一轮日志，metrics 中的 `comm_in_bytes/comm_out_bytes` 为粗略估算（参数数组字节 + 控制负载）。
- GPU 使用：
  - `--device cuda` 或 `--device auto`，必要时用 `--gpu/--gpu_ids/--gpus` 限制可见卡。
- 跑得慢：
  - 将 `SAMPLE_COUNT/BATCH_SIZE/LOCAL_STEPS/ROUNDS` 调小；或者 `--scope reduced`；或用 CPU 快速验证流程。

### 二次开发建议
- 想做稀疏通信：在 strong 端只回传“索引+符号/量化值”，server/weak 端解码重建方向（目前已支持全量方向 blob，可改造为稀疏格式）。
- 想更贴近目标相似度：`generate_instruct_directions_with_R(..., tune_to_target=True)` 会在达到下限后做微调，使余弦更接近设定目标值。

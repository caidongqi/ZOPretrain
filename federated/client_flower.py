import argparse
import time
from typing import List, Tuple, Dict, Any, Optional
import sys
import os
from pathlib import Path
import json

import numpy as np
from datetime import datetime
import csv
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
import base64
import math

import flwr as fl
import os

# 固定本地缓存目录
os.environ.setdefault("HF_HOME", "/data/pc/ZOPretrain/cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/pc/ZOPretrain/cache/hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/pc/ZOPretrain/cache/hf")
# 若完全离线，打开这一行（确保缓存已存在）
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 将项目根目录加入 sys.path，便于从 federated/ 下导入根目录模块
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reproduce_zo_paper import (
    create_model,
    get_trainable_parameters,
    zo_gradient_estimator,
)
from federated.data_utils import build_client_dataloader
from bp_instruct import (
    compute_backprop_gradients,
    generate_instruct_directions,
    generate_instruct_directions_with_R,
)


def tensors_to_numpy(params: List[torch.nn.Parameter]) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for p in params]


def numpy_to_tensors(np_params: List[np.ndarray], params: List[torch.nn.Parameter]) -> None:
    if len(np_params) != len(params):
        raise ValueError(f"Received {len(np_params)} arrays, expected {len(params)}")
    for arr, p in zip(np_params, params):
        if tuple(arr.shape) != tuple(p.data.shape):
            raise ValueError(f"Shape mismatch: got {arr.shape}, expected {tuple(p.data.shape)}")
        p.data = torch.from_numpy(arr).to(p.device)


class ZOFLClient(fl.client.NumPyClient):
    def __init__(
        self,
        *,
        client_id: int,
        num_clients: int,
        mode: str,
        scope: str,
        q: int,
        lr: float,
        local_epochs: int,
        local_steps: Optional[int],
        batch_size: int,
        block_size: int,
        cache_dir: str,
        sample_count: int,
        optimizer_name: str,
        weight_decay: float,
        eps: float,
        betas: Tuple[float, float],
        muon_cautious: bool,
        muon_orthogonal_init: bool,
        muon_hidden_size: int,
        zo_use_optimizer: bool,
        device: str,
        csv_file: Optional[str] = None,
        log_interval: int = 10,
    ) -> None:
        self.client_id = client_id
        self.num_clients = num_clients
        self.mode = mode
        self.scope = scope
        self.q = q
        self.lr = lr
        self.local_epochs = local_epochs
        self.local_steps = local_steps
        self.batch_size = batch_size
        self.block_size = block_size
        self.cache_dir = cache_dir
        self.sample_count = sample_count
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = betas
        self.muon_cautious = muon_cautious
        self.muon_orthogonal_init = muon_orthogonal_init
        self.muon_hidden_size = muon_hidden_size
        self.zo_use_optimizer = zo_use_optimizer
        self.device = device
        self.csv_file = csv_file
        self.log_interval = max(1, int(log_interval))
        self.round_idx = 0
        # 客户端角色：默认 weak，可通过环境变量 FL_CLIENT_ROLE 指定 strong/weak
        self.role = os.environ.get("FL_CLIENT_ROLE", "weak")

        # Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir=os.environ.get("TRANSFORMERS_CACHE", "/data/pc/ZOPretrain/cache/hf"),
            # 离线时设 True；首次需联网下载就设 False
            local_files_only=os.environ.get("TRANSFORMERS_OFFLINE", "") == "1",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = create_model(len(self.tokenizer)).to(self.device)
        # 多卡支持：若可见 GPU 数 > 1，使用 DataParallel
        if self.device == 'cuda' and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.train()

        # Trainable parameters subset
        base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        self.trainable_params: List[torch.nn.Parameter] = get_trainable_parameters(base_model, self.scope)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.trainable_params:
            p.requires_grad = True

        # Optimizer for FO or ZO (when zo_use_optimizer)
        self.optimizer = None
        if self.mode == 'FO' or (self.mode == 'ZO' and self.zo_use_optimizer):
            if self.optimizer_name == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.trainable_params,
                    lr=self.lr,
                    betas=self.betas,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            elif self.optimizer_name == 'muon':
                from optim_muon import AdamW as MuonAdamW
                self.optimizer = MuonAdamW(
                    self.trainable_params,
                    lr=self.lr,
                    betas=self.betas,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                    correct_bias=True,
                    cautious=self.muon_cautious,
                    orthogonal_init=self.muon_orthogonal_init,
                    hidden_size=self.muon_hidden_size,
                    no_deprecation_warning=True,
                )

        # DataLoader for this client partition
        self.trainloader: DataLoader = build_client_dataloader(
            self.tokenizer,
            batch_size=self.batch_size,
            block_size=self.block_size,
            cache_dir=self.cache_dir,
            num_clients=self.num_clients,
            client_id=self.client_id,
            sample_count=self.sample_count,
        )

        self.loss_fn = CrossEntropyLoss()

        # 初始化 CSV 文件（若启用）
        if self.csv_file is not None:
            csv_path = Path(self.csv_file)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not csv_path.exists():
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'server_round', 'epoch', 'step',
                        'mode', 'scope', 'q', 'lr', 'batch_size',
                        'loss', 'grad_norm', 'client_id'
                    ])

    # Flower Hooks
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return tensors_to_numpy(self.trainable_params)

    def set_parameters_from_server(self, parameters: Optional[List[np.ndarray]]) -> None:
        if parameters is None or len(parameters) == 0:
            return
        numpy_to_tensors(parameters, self.trainable_params)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        self.set_parameters_from_server(parameters)
        compute_device = self.trainable_params[0].device if self.trainable_params else torch.device(self.device if self.device == "cuda" else "cpu")

        start_time = time.time()
        total_loss = torch.zeros(1, device=compute_device, dtype=torch.float64)
        step_count = 0
        server_round = int(config.get('server_round', self.round_idx + 1))

        def log_csv_entry(server_round_val: int, epoch_val: int, step_val: int, loss_val: float, grad_val: float) -> None:
            if self.csv_file is None:
                return
            try:
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        int(server_round_val),
                        int(epoch_val),
                        int(step_val),
                        self.mode,
                        self.scope,
                        self.q if self.mode == 'ZO' else 'N/A',
                        self.lr,
                        self.batch_size,
                        float(loss_val),
                        float(grad_val),
                        self.client_id,
                    ])
            except Exception:
                pass

        # 估算输入通信量（参数 + 控制字段）
        try:
            in_param_bytes = sum(int(arr.nbytes) for arr in parameters) if parameters is not None else 0
        except Exception:
            in_param_bytes = 0

        # Instruct：BP 阶段（仅 strong 客户端执行 BP 选择 top-k 种子）
        if bool(config.get('instruct_enable', False)) and str(config.get('instruct_phase', '')) == 'bp_select':
            metrics: Dict[str, Any] = {
                "client_id": self.client_id,
                "round": server_round,
                "mode": self.mode,
                "role": self.role,
                "bp_select": True,
            }
            if str(self.role).lower() != "strong":
                metrics["skipped_bp"] = True
                # 输出通信：返回参数
                out_param_bytes = sum(int(arr.nbytes) for arr in tensors_to_numpy(self.trainable_params))
                metrics["comm_in_bytes"] = in_param_bytes
                metrics["comm_out_bytes"] = out_param_bytes
                log_csv_entry(server_round, 0, 0, 0.0, 0.0)
                self.round_idx = server_round
                return tensors_to_numpy(self.trainable_params), len(self.trainloader.dataset), metrics

            # strong：读取候选种子并评估投影 g^T u（保留以便 Server 可用），并额外生成与 BP 梯度具有指定相似度的方向并发送
            try:
                cand_json = config.get('bp_candidate_seeds_json', '[]')
                candidate_seeds = json.loads(cand_json) if isinstance(cand_json, str) else list(cand_json)
            except Exception:
                candidate_seeds = []
            topk = int(config.get('bp_select_topk', max(1, int(config.get('instruct_zo_dir_count', 1)))))
            # instruct 方向生成配置
            instruct_dir_count = int(config.get('instruct_dir_count', topk))
            instruct_cos_target = float(config.get('instruct_cosine_target', 0.9))
            instruct_dir_method = str(config.get('instruct_dir_method', 'orth')).lower()  # 'orth' or 'r'
            # 抽一小批数据来计算一次 BP 梯度
            bp_inputs = None
            bp_labels = None
            for batch in self.trainloader:
                bp_inputs = batch.to(self.device)
                bp_labels = bp_inputs.clone()
                break
            if bp_inputs is None:
                metrics["skipped_bp"] = True
                out_param_bytes = sum(int(arr.nbytes) for arr in tensors_to_numpy(self.trainable_params))
                metrics["comm_in_bytes"] = in_param_bytes
                metrics["comm_out_bytes"] = out_param_bytes
                log_csv_entry(server_round, 0, 0, 0.0, 0.0)
                self.round_idx = server_round
                return tensors_to_numpy(self.trainable_params), len(self.trainloader.dataset), metrics

            _, bp_grads = compute_backprop_gradients(
                self.model if not isinstance(self.model, torch.nn.DataParallel) else self.model.module,
                self.trainable_params,
                self.loss_fn,
                bp_inputs,
                bp_labels,
            )
            proj_vals: List[Tuple[int, float]] = []
            torch_device = compute_device if isinstance(compute_device, torch.device) else self.device
            for seed in candidate_seeds:
                try:
                    gen = torch.Generator(device=torch_device).manual_seed(int(seed))
                except Exception:
                    gen = torch.Generator().manual_seed(int(seed))
                proj_accum = torch.zeros(1, device=compute_device, dtype=torch.float64)
                for p, g in zip(self.trainable_params, bp_grads):
                    g_dev = g.detach().to(p.device)
                    u = torch.randn(p.data.shape, generator=gen, device=p.device, dtype=p.dtype)
                    proj_accum = proj_accum + torch.sum(g_dev * u).to(device=proj_accum.device, dtype=proj_accum.dtype)
                proj_vals.append((int(seed), float(proj_accum.item())))
            proj_vals.sort(key=lambda x: abs(x[1]), reverse=True)
            top_seeds = [s for s, _ in proj_vals[:topk]]
            top_projs = [v for _, v in proj_vals[:topk]]

            # 生成与 BP 梯度具有余弦相似度 ≈ instruct_cos_target 的方向
            total_norm_sq = torch.zeros(1, device=compute_device, dtype=torch.float64)
            for g in bp_grads:
                g_dev = g.detach().to(total_norm_sq.device)
                total_norm_sq = total_norm_sq + torch.sum(g_dev * g_dev).to(device=total_norm_sq.device, dtype=total_norm_sq.dtype)
            total_norm = torch.sqrt(total_norm_sq.clamp_min(0.0))
            gen_dirs: List[List[torch.Tensor]] = []
            total_norm_value = float(total_norm.item())
            total_norm_sq_value = float(total_norm_sq.item())
            if total_norm_value > 0.0:
                if instruct_dir_method == 'r':
                    dir_iter = generate_instruct_directions_with_R(
                        bp_grads,
                        instruct_dir_count,
                        instruct_cos_target,
                        total_norm_value,
                        total_norm_sq_value,
                        device=self.device,
                    )
                else:
                    dir_iter = generate_instruct_directions(
                        bp_grads,
                        instruct_dir_count,
                        instruct_cos_target,
                        total_norm_value,
                        total_norm_sq_value,
                    )
                if dir_iter is not None:
                    for dirs in dir_iter:
                        gen_dirs.append([d.detach().clone() for d in dirs])
            # 序列化方向（base64），包含每个参数形状
            shapes = [list(p.data.shape) for p in self.trainable_params]
            def _encode_dirs(directions: List[List[torch.Tensor]]) -> str:
                payload: Dict[str, Any] = {
                    "dtype": "float32",
                    "shapes": shapes,
                    "dirs": [],
                }
                for one in directions:
                    per_param_b64: List[str] = []
                    for d in one:
                        arr = d.detach().to('cpu', dtype=torch.float32).numpy()
                        per_param_b64.append(base64.b64encode(arr.tobytes()).decode('ascii'))
                    payload["dirs"].append(per_param_b64)
                return json.dumps(payload)
            dir_blob_json = _encode_dirs(gen_dirs) if len(gen_dirs) > 0 else json.dumps({"dtype":"float32","shapes":shapes,"dirs":[]})

            out_param_bytes = sum(int(arr.nbytes) for arr in tensors_to_numpy(self.trainable_params))
            out_ctrl_bytes = int(4 * len(top_seeds) + 8 * len(top_projs)) + len(dir_blob_json.encode('utf-8'))
            metrics.update({
                "bp_top_seeds_json": json.dumps(top_seeds),
                "bp_top_projs_json": json.dumps(top_projs),
                "bp_candidate_count": len(candidate_seeds),
                "bp_topk": topk,
                "instruct_dir_blob_json": dir_blob_json,
                "instruct_dir_count": len(gen_dirs),
                "instruct_cosine_target": instruct_cos_target,
                "instruct_dir_method": instruct_dir_method,
                "comm_in_bytes": in_param_bytes + int(4 * len(candidate_seeds)),
                "comm_out_bytes": out_param_bytes + out_ctrl_bytes,
            })
            bp_loss_value = 0.0
            try:
                was_training = self.model.training
                if was_training:
                    self.model.eval()
                with torch.no_grad():
                    logits = self.model(bp_inputs).logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = bp_labels[:, 1:].contiguous()
                    bp_loss = self.loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    bp_loss_value = float(bp_loss.item())
                if was_training:
                    self.model.train()
            except Exception:
                bp_loss_value = 0.0
            log_csv_entry(server_round, 0, 1, bp_loss_value, float(total_norm_value))
            self.round_idx = server_round
            return tensors_to_numpy(self.trainable_params), len(self.trainloader.dataset), metrics

        # Server-side ZO path: clients do NOT update params; only report directional derivatives
        if (self.mode == 'ZO') and bool(config.get('zo_server_side', False)):
            # Instruct：ZO 评估阶段仅 weak 客户端执行
            if bool(config.get('instruct_enable', False)) and str(config.get('instruct_phase', '')) == 'zo_eval':
                if str(self.role).lower() != "weak":
                    metrics = {
                        "client_id": self.client_id,
                        "train_time_s": time.time() - start_time,
                        "avg_loss": 0.0,
                        "steps": step_count,
                        "mode": self.mode,
                        "q": self.q if self.mode == 'ZO' else -1,
                        "round": server_round,
                        "lr": self.lr,
                        "skipped_zo_eval": True,
                        "comm_in_bytes": in_param_bytes,
                        "comm_out_bytes": sum(int(arr.nbytes) for arr in tensors_to_numpy(self.trainable_params)),
                    }
                    self.round_idx = server_round
                    return tensors_to_numpy(self.trainable_params), len(self.trainloader.dataset), metrics
            epsilon = float(config.get('zo_epsilon', 1e-4))
            # 优先使用服务端下发的“指令方向” blob；若不存在则回退到 seeds
            dir_blob_json = config.get('instruct_dir_blob_json', None)
            eval_steps = int(config.get('zo_eval_steps', 1))
            provided_dirs: List[List[torch.Tensor]] = []

            def _compute_loss_current() -> Tuple[float, int]:
                was_training = self.model.training
                if was_training:
                    self.model.eval()
                total = torch.zeros(1, device=compute_device, dtype=torch.float64)
                steps_local = 0
                try:
                    with torch.no_grad():
                        for batch in self.trainloader:
                            inputs = batch.to(self.device)
                            labels = inputs.clone()
                            logits = self.model(inputs).logits
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = labels[:, 1:].contiguous()
                            loss = self.loss_fn(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                            )
                            total.add_(loss.detach().to(device=total.device, dtype=total.dtype))
                            steps_local += 1
                            if steps_local >= eval_steps:
                                break
                finally:
                    if was_training:
                        self.model.train()
                return float((total / max(steps_local, 1)).item()), steps_local

            baseline_loss, baseline_steps = _compute_loss_current()
            if dir_blob_json:
                try:
                    blob = json.loads(dir_blob_json)
                    shapes = blob.get("shapes", [])
                    dirs_b64 = blob.get("dirs", [])
                    provided_dirs = []
                    for one in dirs_b64:
                        tensors_one: List[torch.Tensor] = []
                        for shape, b64 in zip(shapes, one):
                            raw = base64.b64decode(b64.encode('ascii'))
                            arr = torch.frombuffer(memoryview(raw), dtype=torch.float32).clone().view(*shape)
                            tensors_one.append(arr.to(self.device))
                        provided_dirs.append(tensors_one)
                except Exception:
                    provided_dirs = []
                    dir_blob_json = None
            if not dir_blob_json:
                # 兼容服务端种子字段（标量限制：JSON 字符串）
                dir_seeds_val = config.get('zo_dir_seeds')
                if isinstance(dir_seeds_val, str):
                    try:
                        dir_seeds = json.loads(dir_seeds_val)
                    except Exception:
                        dir_seeds = []
                elif isinstance(dir_seeds_val, (list, tuple)):
                    dir_seeds = list(dir_seeds_val)
                else:
                    try:
                        dir_seeds = json.loads(config.get('zo_dir_seeds_json', '[]'))
                    except Exception:
                        dir_seeds = []

            def _loss_with_perturb(sign: float, dirs: List[torch.Tensor]) -> float:
                # apply
                for p, u in zip(self.trainable_params, dirs):
                    p.data.add_(sign * epsilon * u.to(p.device))
                try:
                    loss_value, _ = _compute_loss_current()
                    return loss_value
                finally:
                    # revert
                    for p, u in zip(self.trainable_params, dirs):
                        p.data.add_(-sign * epsilon * u.to(p.device))

            g_dir_list: List[float] = []
            incoming_ctrl_bytes = in_param_bytes
            if provided_dirs:
                # 沿“指令方向”前向有限差分
                for dirs in provided_dirs:
                    lp = _loss_with_perturb(+1.0, dirs)
                    lm = _loss_with_perturb(-1.0, dirs)
                    g_dir = (lp - lm) / (2.0 * epsilon)
                    g_dir_list.append(float(g_dir))
                # 统计控制负载
                incoming_ctrl_bytes += len(dir_blob_json.encode('utf-8')) if dir_blob_json else 0
            else:
                # 回退：按 seeds 构建方向
                for seed in dir_seeds:
                    gen = torch.Generator(device=self.device if self.device == 'cuda' else 'cpu').manual_seed(int(seed))
                    dirs: List[torch.Tensor] = []
                    for p in self.trainable_params:
                        u = torch.randn(p.data.shape, generator=gen, device=p.device, dtype=p.dtype)
                        dirs.append(u)
                    lp = _loss_with_perturb(+1.0, dirs)
                    lm = _loss_with_perturb(-1.0, dirs)
                    g_dir = (lp - lm) / (2.0 * epsilon)
                    g_dir_list.append(float(g_dir))
                incoming_ctrl_bytes += int(4 * len(dir_seeds))

            grad_norm_dirs = math.sqrt(sum(g * g for g in g_dir_list)) if g_dir_list else 0.0
            logged_steps = max(baseline_steps, eval_steps * max(1, len(g_dir_list)))
            step_count = max(step_count, logged_steps)
            log_csv_entry(server_round, 0, logged_steps, baseline_loss, grad_norm_dirs)

            metrics = {
                "client_id": self.client_id,
                "train_time_s": time.time() - start_time,
                "avg_loss": 0.0,
                "steps": step_count,
                "mode": self.mode,
                "q": self.q if self.mode == 'ZO' else -1,
                "round": server_round,
                "lr": self.lr,
                "zo_dir_g_json": json.dumps(g_dir_list),
                "comm_in_bytes": incoming_ctrl_bytes,
                "comm_out_bytes": sum(int(arr.nbytes) for arr in tensors_to_numpy(self.trainable_params)) + int(8 * len(g_dir_list)),
            }
            self.round_idx = server_round
            # return parameters unchanged
            return tensors_to_numpy(self.trainable_params), len(self.trainloader.dataset), metrics

        # Local training
        epsilon = 1e-4
        for epoch in range(1, self.local_epochs + 1):
            for batch in self.trainloader:
                inputs = batch.to(self.device)
                labels = inputs.clone()

                if self.mode == 'FO':
                    assert self.optimizer is not None
                    self.optimizer.zero_grad()
                    logits = self.model(inputs).logits               # [B, T, V]
                    shift_logits = logits[:, :-1, :].contiguous()    # [B, T-1, V]
                    shift_labels = labels[:, 1:].contiguous()        # [B, T-1]
                    loss = self.loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    loss.backward()
                    # 计算梯度范数（FO）
                    grad_norm_sq = torch.zeros(1, device=compute_device, dtype=torch.float64)
                    for p in self.trainable_params:
                        if p.grad is not None:
                            g = p.grad.detach()
                            grad_norm_sq = grad_norm_sq + torch.sum(g * g).to(device=grad_norm_sq.device, dtype=grad_norm_sq.dtype)
                    grad_norm = torch.sqrt(grad_norm_sq.clamp_min(0.0))
                    self.optimizer.step()
                else:  # ZO
                    with torch.no_grad():
                        logits = self.model(inputs).logits
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        loss = self.loss_fn(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )

                    grad_paramwise = zo_gradient_estimator(
                        self.model,
                        self.trainable_params,
                        self.loss_fn,
                        inputs,
                        labels,
                        self.q,
                        epsilon,
                        self.device,
                    )

                    # 计算梯度范数（ZO）
                    grad_norm_sq = torch.zeros(1, device=compute_device, dtype=torch.float64)
                    for g in grad_paramwise:
                        if g is not None:
                            gg = g.detach().to(grad_norm_sq.device)
                            grad_norm_sq = grad_norm_sq + torch.sum(gg * gg).to(device=grad_norm_sq.device, dtype=grad_norm_sq.dtype)
                    grad_norm = torch.sqrt(grad_norm_sq.clamp_min(0.0))

                    if self.zo_use_optimizer and self.optimizer is not None:
                        self.optimizer.zero_grad(set_to_none=True)
                        for p, g in zip(self.trainable_params, grad_paramwise):
                            if g is None:
                                continue
                            p.grad = g.to(p.device)
                        self.optimizer.step()
                    else:
                        for p, g in zip(self.trainable_params, grad_paramwise):
                            if g is None:
                                continue
                            p.data -= self.lr * g

                loss_detached = loss.detach()
                total_loss = total_loss + loss_detached.to(device=total_loss.device, dtype=total_loss.dtype)
                step_count += 1

                # 记录 CSV（按步）
                if step_count % self.log_interval == 0:
                    grad_norm_value = float(grad_norm.item()) if grad_norm is not None else 0.0
                    log_csv_entry(server_round, epoch, step_count, float(loss_detached.item()), grad_norm_value)
                if self.local_steps is not None and step_count >= self.local_steps:
                    break
            if self.local_steps is not None and step_count >= self.local_steps:
                break

        metrics = {
            "client_id": self.client_id,
            "train_time_s": time.time() - start_time,
            "avg_loss": float((total_loss / max(step_count, 1)).item()),
            "steps": step_count,
            "mode": self.mode,
            "q": self.q if self.mode == 'ZO' else -1,
            "round": server_round,
            "lr": self.lr,
        }

        # 更新本地轮次索引
        self.round_idx = server_round

        return tensors_to_numpy(self.trainable_params), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        self.set_parameters_from_server(parameters)
        self.model.eval()
        compute_device = self.trainable_params[0].device if self.trainable_params else torch.device(self.device if self.device == "cuda" else "cpu")
        total_loss = torch.zeros(1, device=compute_device, dtype=torch.float64)
        steps = 0
        with torch.no_grad():
            for batch in self.trainloader:
                inputs = batch.to(self.device)
                labels = inputs.clone()
                logits = self.model(inputs).logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = self.loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                total_loss = total_loss + loss.detach().to(device=total_loss.device, dtype=total_loss.dtype)
                steps += 1
                if steps >= 20:  # limit evaluation cost
                    break
        self.model.train()
        return float((total_loss / max(steps, 1)).item()), len(self.trainloader.dataset), {"eval_steps": steps}


def main():
    parser = argparse.ArgumentParser(description="Flower client for ZO/FO training")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--mode", type=str, choices=["FO", "ZO"], default="ZO")
    parser.add_argument("--scope", type=str, choices=["full", "reduced"], default="full")
    # Client-side ZO directions per step (renamed for clarity); keep --q as alias
    parser.add_argument("--client_zo_q", "--q", dest="client_zo_q", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--local_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--sample_count", type=int, default=20000)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "muon"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--muon_cautious", action="store_true")
    parser.add_argument("--muon_orthogonal_init", action="store_true")
    parser.add_argument("--muon_hidden_size", type=int, default=768)
    parser.add_argument("--zo_use_optimizer", action="store_true")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=None, help="当使用 CUDA 时选择单张 GPU 序号，如 0、1。")
    parser.add_argument("--gpu_ids", type=str, default=None, help="以逗号分隔的多卡 GPU 序号，例如 '0,1,2'")
    parser.add_argument("--gpus", type=int, default=None, help="使用的 GPU 卡数（如 2）。未指定 gpu_ids 时生效")
    parser.add_argument("--csv_file", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--role", type=str, choices=["strong", "weak"], default="weak", help="客户端角色：strong 负责 BP 指导，weak 负责 ZO 评估")

    args = parser.parse_args()

    # 在任何 CUDA 检查/使用前设置可见 GPU（若指定）
    if args.device in ("auto", "cuda"):
        if args.gpu_ids:
            ids = ",".join([s.strip() for s in str(args.gpu_ids).split(',') if s.strip() != ""])
            if ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = ids
        elif args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        elif args.gpus is not None and args.gpus > 0:
            ids = ",".join(str(i) for i in range(int(args.gpus)))
            os.environ["CUDA_VISIBLE_DEVICES"] = ids

    # 设备选择与 CUDA 自检
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        try:
            _ = torch.zeros(1, device="cuda")  # 触发 CUDA 上下文初始化
        except Exception:
            device = "cpu"

    # 自动生成实验目录与 CSV 路径
    def _format_lr(x: float) -> str:
        return "%g" % x

    ops_parts = ["zoopt" if args.zo_use_optimizer else "sgd", f"dev{device}"]
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if device == "cuda" and cuda_visible:
        ids_display = cuda_visible.replace(" ", "")
        ops_parts.append(f"gpus{len(ids_display.split(','))}")
        ops_parts.append(f"ids{ids_display}")
    ops = "_".join(ops_parts)

    exp_dir = Path("results") / (
        f"{args.mode}_{args.optimizer}_{args.scope}_n{args.num_clients}_q{args.client_zo_q}_"
        f"lr{_format_lr(args.lr)}_e{args.local_epochs}_bs{args.batch_size}_sc{args.sample_count}_{ops}"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    auto_csv = args.csv_file if args.csv_file is not None else str(exp_dir / f"client{args.client_id}.csv")

    # 保存参数快照
    try:
        with open(exp_dir / f"client{args.client_id}_config.json", 'w') as f:
                json.dump({
                "server": args.server,
                "client_id": args.client_id,
                "num_clients": args.num_clients,
                "mode": args.mode,
                "scope": args.scope,
                    "client_zo_q": args.client_zo_q,
                    "q": args.client_zo_q,
                "lr": args.lr,
                "local_epochs": args.local_epochs,
                "local_steps": args.local_steps,
                "batch_size": args.batch_size,
                "block_size": args.block_size,
                "cache_dir": args.cache_dir,
                "sample_count": args.sample_count,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "eps": args.eps,
                "betas": args.betas,
                "muon_cautious": args.muon_cautious,
                "muon_orthogonal_init": args.muon_orthogonal_init,
                "muon_hidden_size": args.muon_hidden_size,
                "zo_use_optimizer": args.zo_use_optimizer,
                "device": device,
                "gpu": args.gpu,
                "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", None),
                "gpus": args.gpus,
                "csv_file": auto_csv,
                "log_interval": args.log_interval,
                "exp_dir": str(exp_dir),
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 通过环境变量传达客户端角色（供实例内部读取）
    try:
        os.environ["FL_CLIENT_ROLE"] = str(args.role)
    except Exception:
        pass

    client = ZOFLClient(
        client_id=args.client_id,
        num_clients=args.num_clients,
        mode=args.mode,
        scope=args.scope,
        q=args.client_zo_q,
        lr=args.lr,
        local_epochs=args.local_epochs,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        cache_dir=args.cache_dir,
        sample_count=args.sample_count,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        eps=args.eps,
        betas=tuple(args.betas),
        muon_cautious=args.muon_cautious,
        muon_orthogonal_init=args.muon_orthogonal_init,
        muon_hidden_size=args.muon_hidden_size,
        zo_use_optimizer=args.zo_use_optimizer,
        device=device,
        csv_file=auto_csv,
        log_interval=args.log_interval,
    )

    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()



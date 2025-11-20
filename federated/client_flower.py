import argparse
import time
from typing import List, Tuple, Dict, Any, Optional, Sequence
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
os.environ.setdefault("HF_HOME", "cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "cache/hf")
os.environ.setdefault("HF_DATASETS_CACHE", "cache/hf")
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
from federated.sparse_codec import encode_shared_sparse_dirs_from_flats, decode_shared_sparse_dirs_to_numpy


def tensors_to_numpy(params: List[torch.nn.Parameter]) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for p in params]


def tensor_param_bytes(params: Sequence[torch.nn.Parameter]) -> int:
    return sum(int(p.numel() * p.element_size()) for p in params)


def numpy_to_tensors(np_params: List[np.ndarray], params: List[torch.nn.Parameter]) -> None:
    if len(np_params) != len(params):
        raise ValueError(f"Received {len(np_params)} arrays, expected {len(params)}")
    for arr, p in zip(np_params, params):
        if tuple(arr.shape) != tuple(p.data.shape):
            raise ValueError(f"Shape mismatch: got {arr.shape}, expected {tuple(p.data.shape)}")
        p.data = torch.from_numpy(arr).to(p.device)


def resolve_param_device(
    params: List[torch.nn.Parameter],
    fallback: Optional[torch.device] = None,
) -> torch.device:
    if params:
        return params[0].device
    if fallback is not None:
        return fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_runtime_device(device_pref: str) -> Tuple[str, torch.device]:
    prefers_cuda = device_pref != "cpu"
    if torch.cuda.is_available() and prefers_cuda:
        try:
            torch.zeros(1, device="cuda")
            return "cuda", torch.device("cuda")
        except Exception:
            pass
    return "cpu", torch.device("cpu")


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
        self.device_type = device
        self.device = torch.device("cuda" if self.device_type == "cuda" else "cpu")
        self.torch_device = self.device
        self.csv_file = csv_file
        self.log_interval = max(1, int(log_interval))
        self.round_idx = 0
        # 客户端角色：默认 weak，可通过环境变量 FL_CLIENT_ROLE 指定 strong/weak
        self.role = os.environ.get("FL_CLIENT_ROLE", "weak")

        # Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir=os.environ.get("TRANSFORMERS_CACHE", "cache/hf"),
            # 离线时设 True；首次需联网下载就设 False
            local_files_only=os.environ.get("TRANSFORMERS_OFFLINE", "") == "1",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = create_model(len(self.tokenizer)).to(self.device)
        # 多卡支持：若可见 GPU 数 > 1，使用 DataParallel
        if self.device_type == 'cuda' and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.train()
        # 记录 pad id 以统计有效 token 数
        try:
            self.pad_token_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else None
        except Exception:
            self.pad_token_id = None

        # Trainable parameters subset
        base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        self.trainable_params: List[torch.nn.Parameter] = get_trainable_parameters(base_model, self.scope)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.trainable_params:
            p.requires_grad = True
        self.torch_device = resolve_param_device(self.trainable_params, self.torch_device)

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
        self.torch_device = resolve_param_device(self.trainable_params, self.torch_device)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        self.set_parameters_from_server(parameters)
        compute_device = resolve_param_device(self.trainable_params, self.torch_device)
        generator_device = compute_device.type if isinstance(compute_device, torch.device) else "cuda" if "cuda" in str(compute_device) else "cpu"

        # 若服务器下发了稀疏更新（含seed噪声），在本地直接应用（避免下发全量参数）
        try:
            update_json = config.get("server_update_sparse_json", None)
            if update_json:
                decoded = decode_shared_sparse_dirs_to_numpy(str(update_json), apply_noise=True, device=self.device)
                if decoded and len(decoded) > 0:
                    per_param = decoded[0]
                    if len(per_param) == len(self.trainable_params):
                        for p, arr in zip(self.trainable_params, per_param):
                            p.data.add_(torch.from_numpy(arr).to(p.device, dtype=p.dtype))
        except Exception:
            pass

        start_time = time.time()
        total_loss = torch.zeros(1, device=compute_device, dtype=torch.float64)
        step_count = 0
        tokens_this_round = 0
        server_round = int(config.get('server_round', self.round_idx + 1))

        def count_tokens(batch_ids: torch.Tensor) -> int:
            try:
                if self.pad_token_id is not None:
                    return int((batch_ids != self.pad_token_id).sum().item())
                return int(batch_ids.numel())
            except Exception:
                return int(batch_ids.numel()) if hasattr(batch_ids, "numel") else 0

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

        instruct_enabled = bool(config.get('instruct_enable', False))
        phase = str(config.get('instruct_phase', ''))
        strong_action = str(config.get('strong_action', ''))
        weak_action = str(config.get('weak_action', ''))
        role_lower = str(self.role).lower()
        # instruct 模式下仅统计 strong 端 BP 的 token；其余情况按原逻辑
        count_tokens_enabled = (not instruct_enabled) or (instruct_enabled and role_lower == "strong" and strong_action == "bp_grad")

        # 第一轮 strong-only：weak 客户端直接跳过
        if instruct_enabled and phase == 'strong_only' and role_lower != 'strong':
            metrics = {
                "client_id": self.client_id,
                "round": server_round,
                "mode": self.mode,
                "role": self.role,
                "strong_only_skip": True,
                "comm_in_bytes": in_param_bytes,
                "comm_out_bytes": tensor_param_bytes(self.trainable_params),
                "tokens_this_round": int(tokens_this_round),
            }
            try:
                print(f"[Client {self.client_id}][Round {server_round}] phase=strong_only role={self.role} skip", flush=True)
            except Exception:
                pass
            self.round_idx = server_round
            return [], len(self.trainloader.dataset), metrics

        # 强客户端：计算 BP 梯度并生成近似方向，供服务器更新与下一轮弱客户端使用
        if instruct_enabled and strong_action == 'bp_grad' and role_lower == "strong":
            # 弱客户端在 instruct 模式下不做 BP，仅执行后续的 ZO 评估，不应在这里提前返回
            if role_lower != "strong":
                pass
            else:
                metrics: Dict[str, Any] = {
                    "client_id": self.client_id,
                    "round": server_round,
                    "mode": self.mode,
                    "role": self.role,
                    "phase": phase,
                    "strong_bp": True,
                }

            # 读取候选种子并评估投影 g^T u，额外生成与 BP 梯度具有指定相似度的方向
            try:
                cand_json = config.get('bp_candidate_seeds_json', '[]')
                candidate_seeds = json.loads(cand_json) if isinstance(cand_json, str) else list(cand_json)
            except Exception:
                candidate_seeds = []
            topk = int(config.get('bp_select_topk', max(1, int(config.get('instruct_zo_dir_count', 1)))))
            # instruct 方向生成配置
            instruct_dir_count = int(config.get('instruct_dir_count', topk))
            # 兼容：服务端可直接下发 weak 数量 n_weak（覆盖 instruct_dir_count）
            instruct_dir_count = int(config.get('n_weak', instruct_dir_count))
            instruct_cos_target = float(config.get('instruct_cosine_target', 0.9))
            instruct_dir_method = str(config.get('instruct_dir_method', 'orth')).lower()  # 'orth' or 'r'
            noise_scale = float(config.get('instruct_noise_scale', 0.1))
            noise_policy_cfg = str(config.get('instruct_noise_policy', 'zero')).lower()
            sparse_enable = bool(config.get('instruct_sparse_enable', True))
            # 抽一小批数据来计算一次 BP 梯度
            bp_inputs = None
            bp_labels = None
            for batch in self.trainloader:
                bp_inputs = batch.to(self.device)
                bp_labels = bp_inputs.clone()
                break
            if bp_inputs is None:
                metrics["skipped_bp"] = True
                metrics["comm_in_bytes"] = in_param_bytes
                metrics["comm_out_bytes"] = 0
                metrics["tokens_this_round"] = int(tokens_this_round)
                log_csv_entry(server_round, 0, 0, 0.0, 0.0)
                self.round_idx = server_round
                return [], len(self.trainloader.dataset), metrics

            _, bp_grads = compute_backprop_gradients(
                self.model if not isinstance(self.model, torch.nn.DataParallel) else self.model.module,
                self.trainable_params,
                self.loss_fn,
                bp_inputs,
                bp_labels,
            )
            # Instruct 模式：仅统计 BP 使用的一批 token
            try:
                tokens_this_round += count_tokens(bp_inputs)
            except Exception:
                pass
            proj_vals: List[Tuple[int, float]] = []
            for seed in candidate_seeds:
                try:
                    gen = torch.Generator(device=generator_device).manual_seed(int(seed))
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
                # 若启用稀疏并在服务端加噪，避免在客户端构造大规模掩码/能量扫描以降低显存峰值
                sparse_enable = bool(config.get('instruct_sparse_enable', True))
                if sparse_enable:
                    d = int(sum(int(p.data.numel()) for p in self.trainable_params))
                else:
                    # 兼容原逻辑：根据 BP 能量选择高能索引
                    grad_flat = torch.cat([g.detach().flatten().to(self.device) for g in bp_grads])
                    d = int(grad_flat.numel())
                    abs_sq = grad_flat.abs().pow(2)
                    min_similarity = max(min(abs(instruct_cos_target), 0.9999), 0.9)
                    energy_threshold = float((min_similarity ** 2) * total_norm_value * total_norm_value)
                    k = min(64, d)
                    while True:
                        values, indices = torch.topk(abs_sq, k, largest=True)
                        captured = float(values.sum().item())
                        if captured >= energy_threshold or k >= d:
                            break
                        next_k = min(k * 2, d)
                        if next_k == k:
                            break
                        k = next_k
                    threshold_tensor = torch.tensor(energy_threshold, device=values.device, dtype=values.dtype)
                    cumsum_values = torch.cumsum(values, dim=0)
                    effective_rank_idx = torch.searchsorted(cumsum_values, threshold_tensor)
                    effective_rank = int(effective_rank_idx.item()) + 1
                    effective_rank = max(1, min(effective_rank, values.numel()))
                    high_energy_indices = indices[:effective_rank]
                    low_energy_mask = torch.ones(d, dtype=torch.bool, device=grad_flat.device)
                    low_energy_mask[high_energy_indices] = False
                    num_low_energy_dims = int(low_energy_mask.sum().item())

                # 简化：strong 客户端不再生成方向（包括 R 方法），统一由 server 端根据梯度构造
                shapes = [list(p.data.shape) for p in self.trainable_params]

            # 序列化 BP 梯度向量
            grad_payload: Dict[str, Any] = {
                "dtype": "float32",
                "shapes": shapes,
                "grads": [],
            }
            for g in bp_grads:
                arr = g.detach().to('cpu', dtype=torch.float32).numpy()
                grad_payload["grads"].append(base64.b64encode(arr.tobytes()).decode('ascii'))
            grad_blob_json = json.dumps(grad_payload)

            grad_norm_sq = torch.zeros(1, device=compute_device, dtype=torch.float64)
            for g in bp_grads:
                grad_norm_sq = grad_norm_sq + torch.sum(g.detach().to(grad_norm_sq.device, dtype=grad_norm_sq.dtype) * g.detach().to(grad_norm_sq.device, dtype=grad_norm_sq.dtype))
            grad_norm = torch.sqrt(grad_norm_sq.clamp_min(0.0))

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

            # strong 客户端：将本轮 BP loss 与梯度范数写入本地 CSV（只记一条摘要记录）
            try:
                log_csv_entry(
                    server_round_val=server_round,
                    epoch_val=0,
                    step_val=0,
                    loss_val=float(bp_loss_value),
                    grad_val=float(grad_norm.item()),
                )
            except Exception:
                pass

            out_param_bytes = 0
            out_ctrl_bytes = len(grad_blob_json.encode('utf-8'))
            metrics.update({
                "strong_grad_blob_json": grad_blob_json,
                "strong_grad_norm": float(grad_norm.item()),
                "strong_bp_loss": float(bp_loss_value),
                "comm_in_bytes": in_param_bytes + int(4 * len(candidate_seeds)),
                "comm_out_bytes": out_param_bytes + out_ctrl_bytes,
                "tokens_this_round": int(tokens_this_round),
            })
            try:
                print(
                    f"[Client {self.client_id}][Round {server_round}] phase={phase} role={self.role} "
                    f"bp_loss={bp_loss_value:.6f} grad_norm={float(total_norm_value):.6f}",
                    flush=True,
                )
            except Exception:
                pass
            self.round_idx = server_round
            return [], len(self.trainloader.dataset), metrics

        # Server-side ZO path: clients do NOT update params; only report directional derivatives
        if (self.mode == 'ZO') and bool(config.get('zo_server_side', False)):
            # Instruct：joint 阶段 weak 客户端执行 ZO 评估
            if instruct_enabled:
                if role_lower != "weak" or weak_action not in ("zo_eval", "joint", "joint_eval"):
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
                        "phase": phase,
                        "comm_in_bytes": in_param_bytes,
                        "comm_out_bytes": 0,
                    }
                    self.round_idx = server_round
                    return [], len(self.trainloader.dataset), metrics
            epsilon = float(config.get('zo_epsilon', 1e-4))
            # 优先使用服务端下发的“指令方向” blob；若不存在则回退到 seeds
            dir_blob_json = config.get('instruct_dir_blob_json', None)
            eval_steps = int(config.get('zo_eval_steps', 1))
            provided_dirs: List[List[torch.Tensor]] = []

            def _compute_loss_current() -> Tuple[float, int, int]:
                was_training = self.model.training
                if was_training:
                    self.model.eval()
                total = torch.zeros(1, device=compute_device, dtype=torch.float64)
                steps_local = 0
                tokens_used = 0
                try:
                    with torch.no_grad():
                        for batch in self.trainloader:
                            inputs = batch.to(self.device)
                            labels = inputs.clone()
                            tokens_used += count_tokens(inputs)
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
                return float((total / max(steps_local, 1)).item()), steps_local, int(tokens_used)

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
                    # 应用服务端指定的“微小能量噪声”（每个客户端/每个方向使用不同 seed）
                    try:
                        noise_ratio = float(config.get('instruct_noise_energy_ratio', 0.0))
                    except Exception:
                        noise_ratio = 0.0
                    noise_policy = str(config.get('instruct_noise_policy', 'orth'))
                    try:
                        seed_base = int(config.get('instruct_noise_seed', 0))
                    except Exception:
                        seed_base = 0
                    if noise_ratio > 0.0 and provided_dirs:
                        eps_den = 1e-12
                        for j, dirs in enumerate(provided_dirs):
                            # 计算方向范数（合并所有参数）
                            dir_norm_sq = torch.zeros(1, device=compute_device, dtype=torch.float64)
                            for u in dirs:
                                uu = u.to(dir_norm_sq.device, dtype=torch.float64)
                                dir_norm_sq = dir_norm_sq + torch.sum(uu * uu)
                            dir_norm = torch.sqrt(dir_norm_sq.clamp_min(0.0)).item()
                            if dir_norm <= 0.0:
                                continue
                            # 生成噪声并按策略（orth）去除投影，缩放到目标能量，再相加
                            try:
                                gen = torch.Generator(device=self.device.type).manual_seed(int(seed_base ^ (self.client_id * 997) ^ (j * 10007)))
                            except Exception:
                                gen = None
                            # 先生成噪声
                            noise_list: List[torch.Tensor] = []
                            for u in dirs:
                                z = torch.randn(u.shape, generator=gen, device=u.device, dtype=u.dtype) if gen is not None else torch.randn_like(u)
                                noise_list.append(z)
                            if noise_policy == 'orth':
                                # 去除与方向的投影
                                dot_num = torch.zeros(1, device=compute_device, dtype=torch.float64)
                                for z, u in zip(noise_list, dirs):
                                    zz = z.to(dot_num.device, dtype=torch.float64)
                                    uu = u.to(dot_num.device, dtype=torch.float64)
                                    dot_num = dot_num + torch.sum(zz * uu)
                                coeff = float(dot_num.item()) / (dir_norm * dir_norm + eps_den)
                                for i in range(len(noise_list)):
                                    noise_list[i] = noise_list[i] - coeff * dirs[i]
                            # 归一化噪声到目标能量
                            noise_norm_sq = torch.zeros(1, device=compute_device, dtype=torch.float64)
                            for z in noise_list:
                                zz = z.to(noise_norm_sq.device, dtype=torch.float64)
                                noise_norm_sq = noise_norm_sq + torch.sum(zz * zz)
                            noise_norm = float(torch.sqrt(noise_norm_sq.clamp_min(0.0)).item())
                            target_noise_norm = float((noise_ratio ** 0.5) * dir_norm)
                            scale = (target_noise_norm / (noise_norm + eps_den)) if noise_norm > 0.0 else 0.0
                            if scale > 0.0:
                                for i in range(len(dirs)):
                                    dirs[i] = dirs[i] + scale * noise_list[i]
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

            def _loss_with_perturb(sign: float, dirs: List[torch.Tensor]) -> Tuple[float, int]:
                # apply
                for p, u in zip(self.trainable_params, dirs):
                    p.data.add_(sign * epsilon * u.to(p.device))
                try:
                    loss_value, _, tokens_used = _compute_loss_current()
                    return loss_value, int(tokens_used)
                finally:
                    # revert
                    for p, u in zip(self.trainable_params, dirs):
                        p.data.add_(-sign * epsilon * u.to(p.device))

            g_dir_list: List[float] = []
            used_dirs_list: List[List[torch.Tensor]] = []
            incoming_ctrl_bytes = in_param_bytes
            if provided_dirs:
                # 沿“指令方向”前向有限差分
                for dirs in provided_dirs:
                    lp, tok_p = _loss_with_perturb(+1.0, dirs)
                    lm, tok_m = _loss_with_perturb(-1.0, dirs)
                    if count_tokens_enabled:
                        tokens_this_round += int(tok_p + tok_m)
                    g_dir = (lp - lm) / (2.0 * epsilon)
                    g_dir_list.append(float(g_dir))
                    used_dirs_list.append(dirs)
                # 统计控制负载
                incoming_ctrl_bytes += len(dir_blob_json.encode('utf-8')) if dir_blob_json else 0
            else:
                # 回退：按 seeds 构建方向
                for seed in dir_seeds:
                    gen = torch.Generator(device=generator_device).manual_seed(int(seed))
                    dirs: List[torch.Tensor] = []
                    for p in self.trainable_params:
                        u = torch.randn(p.data.shape, generator=gen, device=p.device, dtype=p.dtype)
                        dirs.append(u)
                    lp, tok_p = _loss_with_perturb(+1.0, dirs)
                    lm, tok_m = _loss_with_perturb(-1.0, dirs)
                    if count_tokens_enabled:
                        tokens_this_round += int(tok_p + tok_m)
                    g_dir = (lp - lm) / (2.0 * epsilon)
                    g_dir_list.append(float(g_dir))
                    used_dirs_list.append(dirs)
                incoming_ctrl_bytes += int(4 * len(dir_seeds))

            # 使用方向导数与方向向量构造一次参数级梯度估计，做一次“临时更新→评估→还原”
            if g_dir_list:
                # 聚合本地方向成参数级梯度估计（与服务端重建逻辑一致的均值聚合）
                grad_est_paramwise: List[torch.Tensor] = [torch.zeros_like(p.data) for p in self.trainable_params]
                dir_cnt = max(1, len(used_dirs_list))
                for coeff, dirs in zip(g_dir_list, used_dirs_list):
                    c = float(coeff) / float(dir_cnt)
                    for j, u in enumerate(dirs):
                        grad_est_paramwise[j].add_(c * u.to(device=grad_est_paramwise[j].device, dtype=grad_est_paramwise[j].dtype))

                def _compute_loss_with_temp_step(grad_list: List[torch.Tensor], step_size: float) -> Tuple[float, int, int]:
                    for p, g in zip(self.trainable_params, grad_list):
                        if g is None:
                            continue
                        p.data.add_(-step_size * g.to(p.device, dtype=p.dtype))
                    try:
                        return _compute_loss_current()
                    finally:
                        for p, g in zip(self.trainable_params, grad_list):
                            if g is None:
                                continue
                            p.data.add_(step_size * g.to(p.device, dtype=p.dtype))

                baseline_loss, baseline_steps, baseline_tokens = _compute_loss_with_temp_step(grad_est_paramwise, float(self.lr))
            else:
                # 无有效方向时退回原始评估
                baseline_loss, baseline_steps, baseline_tokens = _compute_loss_current()
            if count_tokens_enabled:
                tokens_this_round += int(baseline_tokens)

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
                "role": self.role,
                "zo_dir_g_json": json.dumps(g_dir_list),
                "comm_in_bytes": incoming_ctrl_bytes,
                "comm_out_bytes": int(8 * len(g_dir_list)),
                "baseline_loss": float(baseline_loss),
                "baseline_steps": int(baseline_steps),
                "weak_grad_norm": float(grad_norm_dirs),
                "tokens_this_round": int(tokens_this_round),
            }
            try:
                print(
                    f"[Client {self.client_id}][Round {server_round}] phase={phase} role={self.role} "
                    f"baseline_loss={baseline_loss:.6f} grad_norm={grad_norm_dirs:.6f}",
                    flush=True,
                )
            except Exception:
                pass
            self.round_idx = server_round
            # return parameters unchanged
            return [], len(self.trainloader.dataset), metrics

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
                    # token 计数（每步一次前向）
                    if count_tokens_enabled:
                        tokens_this_round += count_tokens(inputs)
                    # 计算梯度范数（FO）
                    grad_norm_sq = torch.zeros(1, device=compute_device, dtype=torch.float64)
                    for p in self.trainable_params:
                        if p.grad is not None:
                            g = p.grad.detach()
                            grad_norm_sq = grad_norm_sq + torch.sum(g * g).to(device=grad_norm_sq.device, dtype=grad_norm_sq.dtype)
                    grad_norm = torch.sqrt(grad_norm_sq.clamp_min(0.0))
                    # 弱客户端只计算不更新
                    if self.role != "weak":
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
                    # 近似统计：ZO 每步约 2*q 次前向
                    if count_tokens_enabled:
                        try:
                            tokens_this_round += int(count_tokens(inputs) * max(1, int(2 * self.q)))
                        except Exception:
                            tokens_this_round += count_tokens(inputs)

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
                        # 弱客户端只计算不更新
                        if self.role != "weak":
                            self.optimizer.step()
                    else:
                        # 弱客户端只计算不更新
                        if self.role != "weak":
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
            "tokens_this_round": int(tokens_this_round),
            "mode": self.mode,
            "q": self.q if self.mode == 'ZO' else -1,
            "round": server_round,
            "lr": self.lr,
            "role": self.role,
        }

        # 更新本地轮次索引
        self.round_idx = server_round

        return [], len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        self.set_parameters_from_server(parameters)
        self.model.eval()
        compute_device = resolve_param_device(self.trainable_params, self.torch_device)
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

    device, _ = resolve_runtime_device(args.device)

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



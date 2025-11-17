import argparse
from typing import Optional, Dict, Any, List, Tuple

import json
import os
import sys
from pathlib import Path
import csv
from datetime import datetime
import numpy as np
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import base64
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

# 将项目根目录加入 sys.path，方便导入根目录下的模块（例如 optim_muon）
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bp_instruct import generate_instruct_directions_with_R
from federated.sparse_codec import decode_shared_sparse_dirs_to_numpy, encode_shared_sparse_dirs_from_flats
from federated.data_utils import load_or_build_examples
from reproduce_zo_paper import create_model

# 默认缓存目录（与客户端保持一致），避免重复下载
os.environ.setdefault("HF_HOME", "/data/pc/ZOPretrain/cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/pc/ZOPretrain/cache/hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/pc/ZOPretrain/cache/hf")

def resolve_runtime_device(device_pref: str) -> Tuple[str, torch.device, str]:
    prefers_cuda = device_pref != "cpu"
    if torch.cuda.is_available() and prefers_cuda:
        try:
            torch.zeros(1, device="cuda")
            return "cuda", torch.device("cuda"), "cuda"
        except Exception:
            pass
    return "cpu", torch.device("cpu"), "cpu"


class InstructFLStrategy(fl.server.strategy.FedAvg):
    """Two-stage Instruct strategy:
    - Stage A (bp_select, odd rounds): server broadcasts a candidate seed pool.
      Only strong clients compute BP once and return top-k seeds aligned with BP gradients.
    - Stage B (zo_eval, even rounds): server broadcasts selected seeds to weak clients.
      Weak clients evaluate directional derivatives; server reconstructs gradients and updates.
    """
    def __init__(
        self,
        *,
        fraction_fit: float,
        fraction_evaluate: float,
        min_fit_clients: int,
        min_available_clients: int,
        candidate_pool: int = 128,
        topk: int = 8,
        dir_count: int = 4,
        epsilon: float = 1e-4,
        server_lr: float = 1e-6,
        device: str = "cuda",
        optimizer_name: str = "sgd",
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        betas: Tuple[float, float] = (0.9, 0.999),
        muon_cautious: bool = False,
        muon_orthogonal_init: bool = False,
        muon_hidden_size: int = 768,
        eval_steps: int = 1,
        server_csv_path: Optional[str] = None,
        # 服务器端评估配置（在参数更新后对一个小评估集计算真实 loss）
        server_eval_enable: bool = True,
        server_eval_batch_size: int = 8,
        server_eval_block_size: int = 128,
        server_eval_sample_count: int = 4096,
        server_eval_cache_dir: str = "cache",
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=min_available_clients,
            on_fit_config_fn=self._on_fit_config_fn,
        )
        self.candidate_pool = int(candidate_pool)
        self.topk = int(topk)
        self.dir_count = int(dir_count)
        self.epsilon = float(epsilon)
        self.server_lr = float(server_lr)
        device_str, torch_device, torch_device_str = resolve_runtime_device(device)
        self.device = device_str
        self._torch_device = torch_device
        self._torch_device_str = torch_device_str
        self.optimizer_name = str(optimizer_name).lower()
        self.weight_decay = float(weight_decay)
        self.eps = float(eps)
        self.betas = (float(betas[0]), float(betas[1]))
        self.muon_cautious = bool(muon_cautious)
        self.muon_orthogonal_init = bool(muon_orthogonal_init)
        self.muon_hidden_size = int(muon_hidden_size)
        self.eval_steps = int(eval_steps)
        # Instruct 方向相似度目标（server 也用同一目标生成方向）
        self.instruct_cosine_target: float = 0.9
        # 为 weak 客户端注入的“微小能量噪声”占比（相对方向能量），如 0.01 表示 1%
        self.instruct_noise_energy_ratio: float = 0.01
        # server-side eval
        self.server_eval_enable = bool(server_eval_enable)
        self.server_eval_batch_size = int(server_eval_batch_size)
        self.server_eval_block_size = int(server_eval_block_size)
        self.server_eval_sample_count = int(server_eval_sample_count)
        self.server_eval_cache_dir = str(server_eval_cache_dir)
        self._eval_tokenizer = None
        self._eval_model = None
        self._eval_loss_fn = None
        self._eval_examples = None
        # state
        # 缓存上一轮 strong 客户端提供的近似方向（供下一轮 weak 客户端使用）
        self._available_dir_blob_json: Optional[str] = None
        self._available_dirs: List[List[np.ndarray]] = []
        # 记录当前轮 weak 客户端应使用的方向/随机种子
        self._weak_dirs_in_use: List[List[np.ndarray]] = []
        self._weak_seed_list_in_use: List[int] = []
        # 暂存本轮 strong 客户端新生成的方向（待聚合后更新为 available）
        self._pending_dir_blob_json: Optional[str] = None
        self._pending_dirs: List[List[np.ndarray]] = []
        self._opt = None  # type: ignore[var-annotated]
        self._params = None  # type: ignore[var-annotated]
        self._csv_path: Optional[Path] = Path(server_csv_path).resolve() if server_csv_path else None
        # 下行更新（server->client）缓存：发送稀疏更新（含seed噪声），避免每轮下发全量参数
        self._server_update_sparse_json: Optional[str] = None
        # 简单稀疏率（按维度的比例选 Top-k），0.001 表示取 0.1% 最高幅值分量
        self._downlink_topk_ratio: float = 1
        # 本轮下行参数字节统计（在 configure_fit 中设置，aggregate_fit 中读取写入）
        self._downlink_param_bytes_last: int = 0
        self._csv_fieldnames = [
            "timestamp",
            "round",
            "phase",
            "strong_clients",
            "weak_clients",
            "loss",
            "strong_bp_loss",
            "weak_baseline_loss",
            "strong_grad_norm",
            "weak_grad_norm",
            "final_grad_norm",
            "total_weight",
            "uplink_param_bytes",
            "downlink_param_bytes",
            "tokens",
        ]
        if self._csv_path:
            try:
                self._csv_path.parent.mkdir(parents=True, exist_ok=True)
                if not self._csv_path.exists():
                    with open(self._csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
                        writer.writeheader()
            except Exception:
                self._csv_path = None

    # 捕获服务器当前全局参数（Flower 在每轮调用时会将当前全局参数传入）
    def configure_fit(  # type: ignore[override]
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,  # type: ignore[name-defined]
    ):
        rnd = server_round
        try:
            nds = parameters_to_ndarrays(parameters)
            # 若未初始化或形状变更，则刷新持久参数副本
            need_reinit = False
            if self._params is None or len(nds) == 0:
                need_reinit = True
            elif len(self._params) != len(nds):
                need_reinit = True
            else:
                for p, arr in zip(self._params, nds):
                    if tuple(p.data.shape) != tuple(arr.shape):
                        need_reinit = True
                        break
            if need_reinit and len(nds) > 0:
                tensors = [torch.from_numpy(arr).to(device=self._torch_device, dtype=torch.float32) for arr in nds]
                self._params = [torch.nn.Parameter(t.clone().detach()) for t in tensors]
                self._opt = None
        except Exception:
            pass
        # 自定义：不再下发全量参数，转为仅通过 config 下发“稀疏更新”与控制字段
        cfg = self._on_fit_config_fn(rnd)
        if self._server_update_sparse_json:
            cfg["server_update_sparse_json"] = self._server_update_sparse_json
            try:
                self._downlink_param_bytes_last = int(len(self._server_update_sparse_json.encode("utf-8")))
            except Exception:
                self._downlink_param_bytes_last = 0
        else:
            self._downlink_param_bytes_last = 0
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        fit_list: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]] = []
        for c in clients:
            cfg_one = dict(cfg)
            # 为每个客户端分配不同的噪声 seed
            try:
                cid_int = int("".join(ch for ch in str(c.cid) if ch.isdigit()) or "0")
            except Exception:
                cid_int = 0
            cfg_one["instruct_noise_seed"] = int(((rnd * 1000003) ^ (cid_int * 911)) & 0x7fffffff)
            fit_ins = fl.common.FitIns(fl.common.ndarrays_to_parameters([]), cfg_one)
            fit_list.append((c, fit_ins))
        return fit_list

    def _ensure_eval_ready(self) -> None:
        if not self.server_eval_enable:
            return
        if self._eval_tokenizer is None:
            self._eval_tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",
                cache_dir=os.environ.get("TRANSFORMERS_CACHE", "/data/pc/ZOPretrain/cache/hf"),
                local_files_only=os.environ.get("TRANSFORMERS_OFFLINE", "") == "1",
            )
            if self._eval_tokenizer.pad_token is None:
                self._eval_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self._eval_model is None:
            self._eval_model = create_model(len(self._eval_tokenizer)).to(self._torch_device)
            self._eval_model.eval()
        if self._eval_loss_fn is None:
            self._eval_loss_fn = CrossEntropyLoss()
        if self._eval_examples is None:
            self._eval_examples = load_or_build_examples(
                self._eval_tokenizer,
                block_size=self.server_eval_block_size,
                cache_dir=self.server_eval_cache_dir,
                sample_count=self.server_eval_sample_count,
            )

    @torch.no_grad()
    def _server_eval_loss(self, updated_weights: List[torch.Tensor], eval_steps: int) -> Optional[float]:
        if not self.server_eval_enable:
            return None
        try:
            self._ensure_eval_ready()
            assert self._eval_model is not None and self._eval_loss_fn is not None and self._eval_examples is not None
            # 将更新后的权重加载到评估模型
            model_params = list(self._eval_model.parameters())
            if len(model_params) != len(updated_weights):
                return None
            for p, w in zip(model_params, updated_weights):
                if tuple(p.data.shape) != tuple(w.shape):
                    return None
                p.data.copy_(w.to(p.device, dtype=p.data.dtype))
            # 评估若干步
            steps = 0
            total = torch.zeros(1, device=self._torch_device, dtype=torch.float64)
            bs = max(1, int(self.server_eval_batch_size))
            for i in range(0, min(len(self._eval_examples), bs * max(1, int(eval_steps))), bs):
                batch = self._eval_examples[i:i+bs]
                if not batch:
                    break
                inputs = torch.stack(batch, dim=0).to(self._torch_device)
                labels = inputs.clone()
                logits = self._eval_model(inputs).logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = self._eval_loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                total.add_(loss.detach().to(device=total.device, dtype=total.dtype))
                steps += 1
                if steps >= max(1, int(eval_steps)):
                    break
            return float((total / max(steps, 1)).item())
        except Exception:
            return None

    def _write_csv_row(self, data: Dict[str, Any]) -> None:
        if not self._csv_path:
            return
        try:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
                row = {field: data.get(field, "") for field in self._csv_fieldnames}
                writer.writerow(row)
        except Exception:
            pass

    def _phase(self, rnd: int) -> str:
        # 第一轮仅 strong 客户端参与，其后各轮 strong/weak 同时参与
        return "strong_only" if rnd == 1 else "joint"

    def _make_seeds_pool(self, rnd: int) -> List[int]:
        target_device = self._torch_device
        try:
            gen = torch.Generator(device=self._torch_device_str).manual_seed(20231111 + rnd)
        except Exception:
            gen = torch.Generator().manual_seed(20231111 + rnd)
            target_device = torch.device("cpu")
        seeds = torch.randint(
            low=0,
            high=2**31 - 1,
            size=(max(1, self.candidate_pool),),
            generator=gen,
            device=target_device,
            dtype=torch.int64,
        )
        return [int(x.item()) for x in seeds.to("cpu")]

    def _on_fit_config_fn(self, rnd: int) -> Dict[str, Any]:
        phase = self._phase(rnd)
        cfg: Dict[str, Any] = {
            "server_round": rnd,
            "instruct_enable": True,
            "instruct_phase": phase,
            "strong_action": "bp_grad",
        }
        cfg["bp_candidate_seeds_json"] = json.dumps(self._make_seeds_pool(rnd))
        cfg["bp_select_topk"] = int(self.topk)
        cfg["instruct_dir_count"] = int(self.dir_count)
        cfg["instruct_cosine_target"] = float(self.instruct_cosine_target)
        cfg["instruct_dir_method"] = "r"
        # 兼容字段：下发 weak 方向数量与“微小能量噪声”参数（客户端据此重建噪声）
        cfg["n_weak"] = int(self.dir_count)
        cfg["instruct_noise_energy_ratio"] = float(self.instruct_noise_energy_ratio)
        cfg["instruct_noise_policy"] = "orth"
        cfg["instruct_sparse_enable"] = True

        # 每轮开始前清空 pending，待本轮 strong 回传后再更新
        self._pending_dir_blob_json = None
        self._pending_dirs = []

        if phase == "strong_only":
            cfg["weak_action"] = "skip"
            self._weak_dirs_in_use = []
            self._weak_seed_list_in_use = []
        else:
            cfg["weak_action"] = "zo_eval"
            cfg["zo_server_side"] = True
            cfg["zo_epsilon"] = float(self.epsilon)
            cfg["zo_eval_steps"] = int(self.eval_steps)
            if self._available_dirs:
                # 若已有解码后的方向，但没有现成的密集 JSON，则临时序列化为密集 JSON 下发
                if not self._available_dir_blob_json:
                    shapes = [list(arr.shape) for arr in self._available_dirs[0]]
                    payload = {
                        "dtype": "float32",
                        "shapes": shapes,
                        "dirs": [],
                    }
                    for one in self._available_dirs:
                        per_param_b64 = []
                        for arr in one:
                            a = np.asarray(arr, dtype=np.float32, order="C")
                            per_param_b64.append(base64.b64encode(a.tobytes()).decode("ascii"))
                        payload["dirs"].append(per_param_b64)
                    self._available_dir_blob_json = json.dumps(payload)
                cfg["instruct_dir_blob_json"] = self._available_dir_blob_json
                self._weak_dirs_in_use = self._available_dirs
                self._weak_seed_list_in_use = []
            else:
                seeds = self._make_seeds_pool(max(1, rnd - 1))[: self.dir_count]
                seeds = [int(s) for s in seeds]
                cfg["zo_dir_seeds_json"] = json.dumps(seeds)
                self._weak_dirs_in_use = []
                self._weak_seed_list_in_use = seeds
        return cfg

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:  # type: ignore[override]
        rnd = server_round
        # 若无结果则直接返回当前全局参数
        if len(results) == 0:
            if self._params is not None:
                new_nds_now = [p.data.detach().to("cpu").numpy() for p in self._params]
                # 记录空聚合一行（带上下行参数字节）
                csv_row_empty: Dict[str, Any] = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "round": rnd,
                    "phase": self._phase(rnd),
                    "strong_clients": 0,
                    "weak_clients": 0,
                    "loss": "",
                    "strong_bp_loss": "",
                    "weak_baseline_loss": "",
                    "strong_grad_norm": "",
                    "weak_grad_norm": "",
                    "final_grad_norm": "",
                    "total_weight": 0.0,
                    "uplink_param_bytes": 0,
                    "downlink_param_bytes": int(self._downlink_param_bytes_last),
                }
                self._write_csv_row(csv_row_empty)
                return ndarrays_to_parameters(new_nds_now), {}
            return super().aggregate_fit(rnd, results, failures)

        phase = self._phase(rnd)
        # 使用服务器持久参数作为当前权重；若不存在则尝试从任一返回中获取（兼容旧路径）
        torch_device = self._torch_device
        torch_device_str = self._torch_device_str
        if self._params is not None and len(self._params) > 0:
            w_tensors = [p.data.detach().clone().to(device=torch_device, dtype=torch.float32) for p in self._params]
        else:
            base_params = results[0][1].parameters
            nds = parameters_to_ndarrays(base_params) if base_params is not None else []
            if len(nds) == 0:
                # 仍无法获得参数，跳过更新但写入一行
                self._write_csv_row({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "round": rnd,
                    "phase": phase,
                    "strong_clients": 0,
                    "weak_clients": 0,
                    "loss": "",
                    "strong_bp_loss": "",
                    "weak_baseline_loss": "",
                    "strong_grad_norm": "",
                    "weak_grad_norm": "",
                    "final_grad_norm": "",
                    "total_weight": 0.0,
                })
                # 返回空参数以维持流程
                return None, {"skipped_update": True, "no_server_params": True}
            w_tensors = [torch.from_numpy(arr).to(device=torch_device, dtype=torch.float32) for arr in nds]
            # 同步持久参数
            self._params = [torch.nn.Parameter(w.clone().detach()) for w in w_tensors]
            self._opt = None

        # 汇总通信字节（若客户端上报）
        in_bytes_list: List[float] = []
        out_bytes_list: List[float] = []
        # 汇总“参数”字节：上行（FitRes.parameters）与本轮下行参数（上一阶段 configure_fit 已统计）
        uplink_param_bytes_total = 0
        tokens_total = 0
        for _, res in results:
            m = res.metrics or {}
            in_bytes = float(m.get("comm_in_bytes", 0.0))
            out_bytes = float(m.get("comm_out_bytes", 0.0))
            if in_bytes > 0:
                in_bytes_list.append(in_bytes)
            if out_bytes > 0:
                out_bytes_list.append(out_bytes)
            # tokens 本轮（客户端上报）
            try:
                t = int(m.get("tokens_this_round", 0))
                if t > 0:
                    tokens_total += t
            except Exception:
                pass
            # FitRes.parameters 转 numpy 后求 nbytes
            try:
                if res.parameters is not None:
                    nds_local = parameters_to_ndarrays(res.parameters)
                    uplink_param_bytes_total += int(sum(int(arr.nbytes) for arr in nds_local))
            except Exception:
                pass

        # 统计 strong 梯度
        strong_grad_sums: List[torch.Tensor] = [
            torch.zeros_like(w, device=torch_device, dtype=torch.float64) for w in w_tensors
        ]
        strong_total_weight: float = 0.0
        strong_count = 0
        strong_grad_norm = 0.0
        # 收集每个 strong 客户端的展平梯度向量，用于服务端生成指令方向
        strong_flat_list: List[torch.Tensor] = []

        # 统计 weak 方向导数
        weak_use_dirs = len(self._weak_dirs_in_use) > 0
        weak_dir_len = len(self._weak_dirs_in_use) if weak_use_dirs else len(self._weak_seed_list_in_use)
        weak_dir_sums = torch.zeros(weak_dir_len, device=torch_device, dtype=torch.float64) if weak_dir_len > 0 else None
        weak_dir_weights = torch.zeros(weak_dir_len, device=torch_device, dtype=torch.float64) if weak_dir_len > 0 else None
        weak_total_weight: float = 0.0
        weak_count = 0
        strong_loss_sum = 0.0
        strong_loss_weight = 0.0
        strong_grad_norm_sum = 0.0
        strong_grad_norm_weight = 0.0
        weak_loss_sum = 0.0
        weak_loss_weight = 0.0
        weak_grad_norm_sum = 0.0
        weak_grad_norm_weight = 0.0

        for _, fitres in results:
            metrics = fitres.metrics or {}
            num_ex = float(fitres.num_examples)

            grad_blob_json = metrics.get("strong_grad_blob_json")
            if grad_blob_json:
                try:
                    blob = json.loads(grad_blob_json)
                    shapes = [tuple(x) for x in blob.get("shapes", [])]
                    grads_b64 = blob.get("grads", [])
                    if shapes and grads_b64 and len(shapes) == len(grads_b64) == len(strong_grad_sums):
                        per_grad_list: List[torch.Tensor] = []
                        for idx, (shp, b64) in enumerate(zip(shapes, grads_b64)):
                            raw = base64.b64decode(b64.encode("ascii"))
                            arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(shp)
                            grad_tensor = torch.from_numpy(arr).to(device=torch_device, dtype=torch.float32)
                            strong_grad_sums[idx] += grad_tensor.to(dtype=torch.float64) * num_ex
                            per_grad_list.append(grad_tensor)
                        strong_total_weight += num_ex
                        strong_count += 1
                        with torch.no_grad():
                            grad_norm_sq = torch.tensor(0.0, device=torch_device, dtype=torch.float64)
                            for g in per_grad_list:
                                grad_norm_sq += torch.sum(g.to(dtype=torch.float64) * g.to(dtype=torch.float64))
                            strong_grad_norm = float(torch.sqrt(grad_norm_sq.clamp_min(0.0)).item())
                        # 展平保存（用于后续生成多样方向）
                        try:
                            flat = torch.cat([g.view(-1) for g in per_grad_list]).to(device=torch_device, dtype=torch.float32)
                            strong_flat_list.append(flat)
                        except Exception:
                            pass
                except Exception:
                    pass

            dir_blob_json = metrics.get("instruct_dir_blob_json")
            # 优先处理稀疏 JSON（带共享支撑与seed噪声），必要时再回退到密集 JSON
            sparse_json = metrics.get("instruct_dir_sparse_json")
            if sparse_json and self._pending_dir_blob_json is None:
                # 简化：不再处理客户端稀疏方向，上由服务端从 strong 梯度生成
                self._pending_dir_blob_json = None
                self._pending_dirs = []
            if dir_blob_json and self._pending_dir_blob_json is None:
                try:
                    blob = json.loads(dir_blob_json)
                    shapes = [tuple(x) for x in blob.get("shapes", [])]
                    dirs_b64 = blob.get("dirs", [])
                    if shapes and dirs_b64:
                        restored: List[List[np.ndarray]] = []
                        for one in dirs_b64:
                            per_param: List[np.ndarray] = []
                            for shp, b64 in zip(shapes, one):
                                raw = base64.b64decode(b64.encode("ascii"))
                                arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(shp)
                                per_param.append(arr)
                            restored.append(per_param)
                        if restored:
                            self._pending_dir_blob_json = dir_blob_json
                            self._pending_dirs = restored[: self.dir_count]
                except Exception:
                    self._pending_dir_blob_json = None
                    self._pending_dirs = []

            g_json = metrics.get("zo_dir_g_json")
            if g_json and weak_dir_len > 0:
                try:
                    g_list = json.loads(g_json)
                except Exception:
                    g_list = []
                trimmed = g_list[:weak_dir_len]
                if trimmed:
                    g_tensor = torch.tensor(trimmed, device=torch_device, dtype=torch.float64)
                    effective_len = g_tensor.numel()
                    if weak_dir_sums is not None and weak_dir_weights is not None:
                        weak_dir_sums[:effective_len] += g_tensor * num_ex
                        weak_dir_weights[:effective_len] += num_ex
                        weak_total_weight += num_ex
                        weak_count += 1

            strong_bp_loss_val = metrics.get("strong_bp_loss")
            if strong_bp_loss_val is not None:
                strong_loss_sum += float(strong_bp_loss_val) * num_ex
                strong_loss_weight += num_ex
            strong_grad_norm_val = metrics.get("strong_grad_norm")
            if strong_grad_norm_val is not None:
                strong_grad_norm_sum += float(strong_grad_norm_val) * num_ex
                strong_grad_norm_weight += num_ex

            weak_baseline_loss_val = metrics.get("baseline_loss")
            if weak_baseline_loss_val is not None:
                weak_loss_sum += float(weak_baseline_loss_val) * num_ex
                weak_loss_weight += num_ex
            weak_grad_norm_val = metrics.get("weak_grad_norm")
            if weak_grad_norm_val is not None:
                weak_grad_norm_sum += float(weak_grad_norm_val) * num_ex
                weak_grad_norm_weight += num_ex

        total_weight = strong_total_weight + weak_total_weight
        if total_weight <= 0.0:
            # 无有效梯度，退化为透传
            avg_c2s = float(torch.tensor(out_bytes_list, device=torch_device, dtype=torch.float64).mean().item()) if out_bytes_list else 0.0
            avg_s2c = float(torch.tensor(in_bytes_list, device=torch_device, dtype=torch.float64).mean().item()) if in_bytes_list else 0.0
            # 也写入一行 CSV（loss 留空），保证每轮都有记录
            csv_row_empty: Dict[str, Any] = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "round": rnd,
                "phase": phase,
                "strong_clients": strong_count,
                "weak_clients": weak_count,
                "loss": "",
                "strong_bp_loss": "",
                "weak_baseline_loss": "",
                "strong_grad_norm": "",
                "weak_grad_norm": "",
                "final_grad_norm": "",
                "total_weight": 0.0,
                "uplink_param_bytes": int(uplink_param_bytes_total),
                "downlink_param_bytes": int(self._downlink_param_bytes_last),
                "tokens": int(tokens_total),
            }
            self._write_csv_row(csv_row_empty)
            metrics: Dict[str, fl.common.Scalar] = {
                "instruct_phase": phase,
                "strong_clients": strong_count,
                "weak_clients": weak_count,
                "avg_comm_client_to_server_bytes": avg_c2s,
                "avg_comm_server_to_client_bytes": avg_s2c,
                "skipped_update": True,
                "uplink_param_bytes_total": int(uplink_param_bytes_total),
                "downlink_param_bytes_total": int(self._downlink_param_bytes_last),
                "tokens_total": int(tokens_total),
            }
            # 返回当前全局参数：优先使用服务器持久参数；否则从当前权重张量导出
            if self._params is not None:
                nds_now = [p.data.detach().to("cpu").numpy() for p in self._params]
            else:
                nds_now = [w.detach().to("cpu").numpy() for w in w_tensors]
            return ndarrays_to_parameters(nds_now), metrics

        # 计算 weak 方向对应的梯度平均
        weak_grad_avg: List[torch.Tensor] = [
            torch.zeros_like(w, device=torch_device, dtype=torch.float64) for w in w_tensors
        ]
        if weak_total_weight > 0.0 and weak_dir_len > 0 and weak_dir_sums is not None and weak_dir_weights is not None:
            dir_means_tensor = torch.zeros(weak_dir_len, device=torch_device, dtype=torch.float64)
            valid_mask = weak_dir_weights > 0
            if bool(valid_mask.any().item()):
                dir_means_tensor[valid_mask] = weak_dir_sums[valid_mask] / weak_dir_weights[valid_mask]
            dir_means = dir_means_tensor.tolist()
            dir_cnt = max(1, weak_dir_len)
            for i, g_bar in enumerate(dir_means):
                if weak_use_dirs:
                    one = self._weak_dirs_in_use[i]
                    for idx, arr in enumerate(one):
                        u = torch.from_numpy(arr.astype(np.float32)).to(device=torch_device, dtype=torch.float32)
                        weak_grad_avg[idx] += (float(g_bar) * u.to(dtype=torch.float64)) / dir_cnt
                else:
                    seed = int(self._weak_seed_list_in_use[i])
                    gen = torch.Generator(device=torch_device_str).manual_seed(seed)
                    for idx, w in enumerate(w_tensors):
                        u = torch.randn(w.shape, generator=gen, device=w.device, dtype=w.dtype)
                        weak_grad_avg[idx] += (float(g_bar) * u.to(dtype=torch.float64)) / dir_cnt
        else:
            weak_grad_avg = [torch.zeros_like(w, device=torch_device, dtype=torch.float64) for w in w_tensors]

        # 汇总 strong + weak 梯度
        final_grad_tensors: List[torch.Tensor] = []
        for idx, w in enumerate(w_tensors):
            strong_sum = strong_grad_sums[idx]
            weak_sum = weak_grad_avg[idx] * weak_total_weight
            total_sum = strong_sum + weak_sum
            avg_grad = total_sum / total_weight
            final_grad_tensors.append(avg_grad.to(dtype=torch.float32))

        strong_loss_mean = (strong_loss_sum / strong_loss_weight) if strong_loss_weight > 0 else None
        weak_loss_mean = (weak_loss_sum / weak_loss_weight) if weak_loss_weight > 0 else None
        strong_grad_norm_mean = (strong_grad_norm_sum / strong_grad_norm_weight) if strong_grad_norm_weight > 0 else None
        weak_grad_norm_mean = (weak_grad_norm_sum / weak_grad_norm_weight) if weak_grad_norm_weight > 0 else None

        opt_name = self.optimizer_name
        if opt_name == "sgd":
            new_w_tensors = [w - self.server_lr * g for (w, g) in zip(w_tensors, final_grad_tensors)]
            new_nds = [t.detach().to("cpu").numpy() for t in new_w_tensors]
        else:
            need_reinit = False
            if self._params is None or len(self._params) != len(w_tensors):
                need_reinit = True
            else:
                for p, w in zip(self._params, w_tensors):
                    if tuple(p.data.shape) != tuple(w.shape):
                        need_reinit = True
                        break
            if need_reinit:
                self._params = [torch.nn.Parameter(w.clone().detach()) for w in w_tensors]
                self._opt = None
            assert self._params is not None
            for p, w in zip(self._params, w_tensors):
                p.data.copy_(w)
            if self._opt is None:
                if opt_name == "adam":
                    self._opt = torch.optim.Adam(
                        self._params, lr=self.server_lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay
                    )
                elif opt_name == "muon":
                    from optim_muon import AdamW as MuonAdamW  # 延迟导入
                    self._opt = MuonAdamW(
                        self._params,
                        lr=self.server_lr,
                        betas=self.betas,
                        eps=self.eps,
                        weight_decay=self.weight_decay,
                        correct_bias=True,
                        cautious=self.muon_cautious,
                        orthogonal_init=self.muon_orthogonal_init,
                        hidden_size=self.muon_hidden_size,
                        no_deprecation_warning=True,
                    )
                else:
                    raise ValueError(f"Unknown optimizer for instruct ZO: {opt_name}")
            assert self._opt is not None
            self._opt.zero_grad(set_to_none=True)
            for p, g in zip(self._params, final_grad_tensors):
                p.grad = g.clone().to(p.device)
            self._opt.step()
            new_nds = [p.data.detach().to("cpu").numpy() for p in self._params]

        # 更新供下一轮使用的方向
        # 优先：使用服务端根据 strong 梯度用“R 方法”生成方向；若无，则使用 pending（来自客户端密集方向）
        server_generated_dirs: List[List[np.ndarray]] = []
        try:
            # 聚合 strong 平均梯度（逐参数）
            if strong_total_weight > 0.0:
                strong_avg_grads = [ (g_sum / strong_total_weight).to(device=torch_device, dtype=torch.float32) for g_sum in strong_grad_sums ]
            else:
                strong_avg_grads = []
            # 若 strong 不可用，则可回退到最终梯度估计
            if not strong_avg_grads:
                strong_avg_grads = [g.detach().clone().to(device=torch_device, dtype=torch.float32) for g in final_grad_tensors]
            # 计算范数
            flat = torch.cat([t.view(-1) for t in strong_avg_grads]).to(device=torch_device, dtype=torch.float32)
            total_norm_sq_val = float(torch.sum(flat * flat).item())
            total_norm_val = float(torch.sqrt(torch.tensor(total_norm_sq_val, device=torch_device, dtype=torch.float32)).item())
            if total_norm_val > 0.0 and self.dir_count > 0:
                seed_val = int((rnd * 1000003) ^ 0x51F15EED) & 0x7fffffff
                dir_iter = generate_instruct_directions_with_R(
                    strong_avg_grads,
                    self.dir_count,
                    self.instruct_cosine_target,
                    total_norm_val,
                    total_norm_sq=total_norm_sq_val,
                    device=torch_device,
                    min_cos_floor=None,
                    seed=seed_val,
                    max_rank=None,
                    tune_to_target=True,
                    use_float32=True,
                )
                if dir_iter is not None:
                    for dirs in dir_iter:
                        per_param_np: List[np.ndarray] = []
                        for d in dirs:
                            per_param_np.append(d.detach().to("cpu", dtype=torch.float32).numpy())
                        server_generated_dirs.append(per_param_np)
        except Exception:
            server_generated_dirs = []

        if server_generated_dirs:
            self._available_dirs = server_generated_dirs
            self._available_dir_blob_json = None  # 需要时在下发时临时序列化
            self._pending_dirs = []
            self._pending_dir_blob_json = None
        elif self._pending_dirs:
            self._available_dirs = self._pending_dirs[: self.dir_count]
            self._available_dir_blob_json = self._pending_dir_blob_json  # 可能为 None，后续按需再临时序列化
        else:
            self._available_dirs = []
            self._available_dir_blob_json = None
        self._weak_dirs_in_use = []
        self._weak_seed_list_in_use = []

        # 计算并编码“下发给客户端的稀疏更新”（基于最终梯度张量），以减少下行带宽
        try:
            # 形成一次性展平向量（在设备上操作）
            update_tensors = [(-self.server_lr) * g.to(device=torch_device, dtype=torch.float32) for g in final_grad_tensors]
            flat_update = torch.cat([t.view(-1) for t in update_tensors]).to(torch_device)
            d = int(flat_update.numel())
            k = int(max(1, min(d, int(self._downlink_topk_ratio * d))))
            if 0 < k < d:
                abs_vals = torch.abs(flat_update)
                idx = torch.topk(abs_vals, k, largest=True, sorted=False).indices
                sparse_flat = torch.zeros_like(flat_update)
                sparse_flat.index_copy_(0, idx, flat_update.index_select(0, idx))
            else:
                sparse_flat = flat_update
            shapes = [list(w.shape) for w in w_tensors]
            # 噪声参数：使用与 Instruct 相同字段的默认值
            noise_policy = "zero"
            noise_alpha = 0.1
            try:
                seed = int((rnd * 1000003) ^ 0x5A5A5A5A)
            except Exception:
                seed = int(rnd) & 0x7fffffff
            total_norm = float(torch.linalg.vector_norm(sparse_flat.detach()).item())
            payload = encode_shared_sparse_dirs_from_flats(
                [sparse_flat],
                d,
                shapes,
                total_norm,
                noise_seed=seed,
                noise_policy=noise_policy,
                alpha=noise_alpha,
                device=torch_device,
            )
            self._server_update_sparse_json = payload
        except Exception:
            self._server_update_sparse_json = None

        avg_c2s = float(torch.tensor(out_bytes_list, device=torch_device, dtype=torch.float64).mean().item()) if out_bytes_list else 0.0
        avg_s2c = float(torch.tensor(in_bytes_list, device=torch_device, dtype=torch.float64).mean().item()) if in_bytes_list else 0.0
        with torch.no_grad():
            final_grad_norm_sq = torch.tensor(0.0, device=torch_device, dtype=torch.float64)
            for g in final_grad_tensors:
                final_grad_norm_sq += torch.sum(g.to(dtype=torch.float64) * g.to(dtype=torch.float64))
            final_grad_norm = float(torch.sqrt(final_grad_norm_sq.clamp_min(0.0)).item())
        metrics: Dict[str, fl.common.Scalar] = {
            "instruct_phase": phase,
            "strong_clients": strong_count,
            "weak_clients": weak_count,
            "zo_optimizer": self.optimizer_name,
            "zo_lr": float(self.server_lr),
            "avg_comm_client_to_server_bytes": avg_c2s,
            "avg_comm_server_to_client_bytes": avg_s2c,
            "total_weight": float(total_weight),
            "final_grad_norm": float(final_grad_norm),
            "uplink_param_bytes_total": int(uplink_param_bytes_total),
            "downlink_param_bytes_total": int(self._downlink_param_bytes_last),
            "tokens_total": int(tokens_total),
        }
        if strong_grad_norm_mean is not None:
            metrics["strong_grad_norm"] = float(strong_grad_norm_mean)
        if weak_grad_norm_mean is not None:
            metrics["weak_grad_norm"] = float(weak_grad_norm_mean)
        if strong_loss_mean is not None:
            metrics["strong_bp_loss"] = float(strong_loss_mean)
        if weak_loss_mean is not None:
            metrics["weak_baseline_loss"] = float(weak_loss_mean)

        # 服务器端真实评估：在参数更新后于一个小评估集上计算 loss
        server_eval_loss: Optional[float] = None
        if self.server_eval_enable:
            if self.optimizer_name == "sgd":
                updated_w_for_eval = [t.detach().clone().to(self._torch_device, dtype=torch.float32) for t in new_w_tensors]
            else:
                assert self._params is not None
                updated_w_for_eval = [p.data.detach().clone().to(self._torch_device, dtype=torch.float32) for p in self._params]
            server_eval_loss = self._server_eval_loss(updated_w_for_eval, self.eval_steps)
            if server_eval_loss is not None:
                metrics["server_eval_loss"] = float(server_eval_loss)

        # 定义“loss”列：优先使用 server_eval_loss；否则回退到 weak/strong 的客户端上报均值
        loss_value = server_eval_loss
        if loss_value is None:
            if weak_loss_mean is not None:
                loss_value = float(weak_loss_mean)
            elif strong_loss_mean is not None:
                loss_value = float(strong_loss_mean)

        csv_row: Dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "round": rnd,
            "phase": phase,
            "strong_clients": strong_count,
            "weak_clients": weak_count,
            "loss": "" if loss_value is None else loss_value,
            "strong_bp_loss": "" if strong_loss_mean is None else float(strong_loss_mean),
            "weak_baseline_loss": "" if weak_loss_mean is None else float(weak_loss_mean),
            "strong_grad_norm": "" if strong_grad_norm_mean is None else float(strong_grad_norm_mean),
            "weak_grad_norm": "" if weak_grad_norm_mean is None else float(weak_grad_norm_mean),
            "final_grad_norm": float(final_grad_norm),
            "total_weight": float(total_weight),
            "uplink_param_bytes": int(uplink_param_bytes_total),
            "downlink_param_bytes": int(self._downlink_param_bytes_last),
            "tokens": int(tokens_total),
        }
        self._write_csv_row(csv_row)
        try:
            print(
                f"[Instruct][Round {rnd}] phase={phase} strong={strong_count} weak={weak_count} "
                  f"avg_c->s={metrics['avg_comm_client_to_server_bytes']:.1f}B "
                f"avg_s->c={metrics['avg_comm_server_to_client_bytes']:.1f}B "
                f"up_param={int(uplink_param_bytes_total)}B down_param={int(self._downlink_param_bytes_last)}B "
                f"tokens={int(tokens_total)} "
                f"||g_strong||={metrics.get('strong_grad_norm', 0.0):.3e} ||g_weak||={metrics.get('weak_grad_norm', 0.0):.3e} "
                f"||g_final||={metrics['final_grad_norm']:.3e}",
                flush=True,
            )
        except Exception:
            pass
        return ndarrays_to_parameters(new_nds), metrics


class ServerSideZOStrategy(fl.server.strategy.FedAvg):
    """Server-side ZO updater using seeds and client-computed directional derivatives.

    - Sends seeds and epsilon to clients via on_fit_config_fn
    - Expects clients to return metrics with key 'zo_dir_g_json' containing a JSON list of
      directional derivatives (one per seed): g^T u_i ≈ (f(x+e u_i) - f(x-e u_i))/(2e)
    - Reconstructs directions with torch.randn using the same seeds and current parameter shapes
    - Forms gradient estimate: sum_i (bar_g_i * u_i) / dir_count, optionally weighted by num_examples
    - Updates global parameters with chosen optimizer (sgd/adam/muon)
    """

    def __init__(
        self,
        *,
        fraction_fit: float,
        fraction_evaluate: float,
        min_fit_clients: int,
        min_available_clients: int,
        dir_count: int = 1,
        epsilon: float = 1e-4,
        server_lr: float = 1e-6,
        device: str = "cuda",
        optimizer_name: str = "sgd",
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        betas: Tuple[float, float] = (0.9, 0.999),
        muon_cautious: bool = False,
        muon_orthogonal_init: bool = False,
        muon_hidden_size: int = 768,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=min_available_clients,
            on_fit_config_fn=self._on_fit_config_fn,
        )
        self.dir_count = int(dir_count)
        self.epsilon = float(epsilon)
        self.server_lr = float(server_lr)
        device_str, torch_device, torch_device_str = resolve_runtime_device(device)
        self.device = device_str
        self._torch_device = torch_device
        self._torch_device_str = torch_device_str
        # Optimizer config (server-side ZO)
        self.optimizer_name = str(optimizer_name).lower()
        self.weight_decay = float(weight_decay)
        self.eps = float(eps)
        self.betas = (float(betas[0]), float(betas[1]))
        self.muon_cautious = bool(muon_cautious)
        self.muon_orthogonal_init = bool(muon_orthogonal_init)
        self.muon_hidden_size = int(muon_hidden_size)
        # Persistent optimizer & parameter tensors for server-side updates
        self._opt = None  # type: ignore[var-annotated]
        self._params = None  # type: ignore[var-annotated]
    # 捕获服务器当前全局参数（在每轮开头保存，用于当客户端不回传参数时作为基准）
    def configure_fit(  # type: ignore[override]
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,  # type: ignore[name-defined]
    ):
        rnd = server_round
        try:
            nds = parameters_to_ndarrays(parameters)
            need_reinit = False
            if self._params is None or len(nds) == 0:
                need_reinit = True
            elif len(self._params) != len(nds):
                need_reinit = True
            else:
                for p, arr in zip(self._params, nds):
                    if tuple(p.data.shape) != tuple(arr.shape):
                        need_reinit = True
                        break
            if need_reinit and len(nds) > 0:
                tensors = [torch.from_numpy(arr).to(device=self._torch_device, dtype=torch.float32) for arr in nds]
                self._params = [torch.nn.Parameter(t.clone().detach()) for t in tensors]
                self._opt = None
        except Exception:
            pass
        return super().configure_fit(rnd, parameters, client_manager)

    def _make_dir_seeds(self, rnd: int) -> List[int]:
        target_device = self._torch_device
        try:
            gen = torch.Generator(device=self._torch_device_str).manual_seed(12345 + rnd)
        except Exception:
            gen = torch.Generator().manual_seed(12345 + rnd)
            target_device = torch.device("cpu")
        seeds = torch.randint(
            low=0,
            high=2**31 - 1,
            size=(self.dir_count,),
            generator=gen,
            device=target_device,
            dtype=torch.int64,
        )
        return [int(x.item()) for x in seeds.to("cpu")]

    def _on_fit_config_fn(self, rnd: int) -> Dict[str, Any]:
        cfg = {"server_round": rnd}
        # 指示客户端启用 server-side ZO 汇报（仅使用标量字段；列表改为 JSON 字符串）
        cfg["zo_server_side"] = True
        cfg["zo_dir_seeds_json"] = json.dumps(self._make_dir_seeds(rnd))
        cfg["zo_epsilon"] = float(self.epsilon)
        cfg["zo_eval_steps"] = 1  # 单步评估以节省通信/计算
        return cfg

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:  # type: ignore[override]
        rnd = server_round
        # 若无结果或失败，返回当前全局参数以维持流程
        if len(results) == 0:
            if self._params is not None:
                new_nds_now = [p.data.detach().to("cpu").numpy() for p in self._params]
                return ndarrays_to_parameters(new_nds_now), {}
            return super().aggregate_fit(rnd, results, failures)

        # 从第一个返回中获取当前参数形状
        first_params = results[0][1].parameters
        if first_params is not None:
            nds = parameters_to_ndarrays(first_params)
        else:
            nds = []
        if len(nds) == 0:
            # 若客户端不回传参数，使用服务器持久参数
            if self._params is None:
                return super().aggregate_fit(rnd, results, failures)
            param_shapes = [tuple(p.data.shape) for p in self._params]
            w_tensors = [p.data.detach().clone().to(device=self._torch_device, dtype=torch.float32) for p in self._params]
        else:
            param_shapes = [arr.shape for arr in nds]
            w_tensors = [torch.from_numpy(arr).to(device=self._torch_device, dtype=torch.float32) for arr in nds]
        torch_device = self._torch_device
        torch_device_str = self._torch_device_str

        # 聚合每个方向的标量导数（按样本数加权）
        seed_list = self._make_dir_seeds(rnd)
        dir_sums = torch.zeros(len(seed_list), device=torch_device, dtype=torch.float64)
        dir_weights = torch.zeros(len(seed_list), device=torch_device, dtype=torch.float64)
        for _, fitres in results:
            m = fitres.metrics or {}
            g_json = m.get("zo_dir_g_json")
            if not g_json:
                continue
            try:
                g_list = json.loads(g_json)
            except Exception:
                continue
            num_ex = float(fitres.num_examples)
            trimmed = g_list[:len(seed_list)]
            if len(trimmed) == 0:
                continue
            g_tensor = torch.tensor(trimmed, device=torch_device, dtype=torch.float64)
            effective_len = g_tensor.numel()
            dir_sums[:effective_len] += g_tensor * num_ex
            dir_weights[:effective_len] += num_ex
        # 计算每个方向的平均导数
        dir_means_tensor = torch.zeros(len(seed_list), device=torch_device, dtype=torch.float64)
        valid_mask = dir_weights > 0
        if bool(valid_mask.any().item()):
            dir_means_tensor[valid_mask] = dir_sums[valid_mask] / dir_weights[valid_mask]
        dir_means = dir_means_tensor.tolist()

        # 重建方向并形成全量梯度估计，之后用所选优化器应用更新
        grad_tensors = [torch.zeros_like(w, device=torch_device) for w in w_tensors]
        dir_cnt = max(1, len(seed_list))
        for seed, g_bar in zip(seed_list, dir_means):
            gen = torch.Generator(device=torch_device_str).manual_seed(int(seed))
            for idx, w in enumerate(w_tensors):
                u = torch.randn(w.shape, generator=gen, device=w.device, dtype=w.dtype)
                grad_tensors[idx] = grad_tensors[idx] + (float(g_bar) * u) / dir_cnt

        # 应用优化器（默认 sgd）
        opt_name = self.optimizer_name
        if opt_name == "sgd":
            new_w_tensors = [ w - self.server_lr * g for (w, g) in zip(w_tensors, grad_tensors) ]
            new_nds = [ t.detach().to("cpu").numpy() for t in new_w_tensors ]
            # 同步持久参数
            self._params = [torch.nn.Parameter(w.clone().detach()) for w in new_w_tensors]
        else:
            # 初始化/同步持久参数与优化器
            need_reinit = False
            if self._params is None or len(self._params) != len(w_tensors):
                need_reinit = True
            else:
                for p, w in zip(self._params, w_tensors):
                    if tuple(p.data.shape) != tuple(w.shape):
                        need_reinit = True
                        break

            if need_reinit:
                self._params = [torch.nn.Parameter(w.clone().detach()) for w in w_tensors]
                self._opt = None

            # 同步当前权重到持久参数容器
            assert self._params is not None
            for p, w in zip(self._params, w_tensors):
                p.data.copy_(w)

            # 创建优化器
            if self._opt is None:
                if opt_name == "adam":
                    self._opt = torch.optim.Adam(
                        self._params, lr=self.server_lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay
                    )
                elif opt_name == "muon":
                    from optim_muon import AdamW as MuonAdamW  # 延迟导入
                    self._opt = MuonAdamW(
                        self._params,
                        lr=self.server_lr,
                        betas=self.betas,
                        eps=self.eps,
                        weight_decay=self.weight_decay,
                        correct_bias=True,
                        cautious=self.muon_cautious,
                        orthogonal_init=self.muon_orthogonal_init,
                        hidden_size=self.muon_hidden_size,
                        no_deprecation_warning=True,
                    )
                else:
                    raise ValueError(f"Unknown optimizer for server-side ZO: {opt_name}")

            # 应用梯度并更新
            assert self._opt is not None
            self._opt.zero_grad(set_to_none=True)
            for p, g in zip(self._params, grad_tensors):
                p.grad = g.clone().to(p.device)
            self._opt.step()

            # 导出到 numpy
            new_nds = [ p.data.detach().to("cpu").numpy() for p in self._params ]
        # 同步持久参数（如果是 sgd 分支，上面已同步；adam/muon 已在 _params 内）
        new_params = ndarrays_to_parameters(new_nds)

        metrics: Dict[str, fl.common.Scalar] = {
            "server_side_zo": True,
            "dir_count": len(seed_list),
            "zo_optimizer": self.optimizer_name,
            "zo_lr": float(self.server_lr),
        }
        return new_params, metrics


def main():
    parser = argparse.ArgumentParser(description="Flower server for ZO/FO federated learning")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min_fit_clients", type=int, default=2)
    parser.add_argument("--min_available_clients", type=int, default=2)
    parser.add_argument("--fraction_fit", type=float, default=1.0)
    parser.add_argument("--fraction_evaluate", type=float, default=0.0)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=None, help="当使用 CUDA 时选择单张 GPU 序号，如 0、1。")
    parser.add_argument("--gpu_ids", type=str, default=None, help="以逗号分隔的多卡 GPU 序号，例如 '0,1,2'")
    parser.add_argument("--gpus", type=int, default=None, help="使用的 GPU 卡数（如 2）。未指定 gpu_ids 时生效")
    # Instruct mode
    parser.add_argument("--instruct_enable", action="store_true", help="Enable two-stage Instruct FL (strong/weak clients)")
    parser.add_argument("--instruct_candidate_pool", type=int, default=128, help="BP 阶段候选随机方向的数量（通过种子下发）")
    parser.add_argument("--instruct_topk", type=int, default=8, help="strong 客户端在 BP 阶段回传的 top-k 种子数")
    parser.add_argument("--instruct_dir_count", type=int, default=4, help="ZO 阶段最终使用的方向数量")
    parser.add_argument("--instruct_eval_steps", type=int, default=1, help="weak 客户端 ZO 评估每轮使用的 batch 步数")
    parser.add_argument("--instruct_server_csv", type=str, default="results/instruct_server_metrics.csv", help="服务端记录 Instruct 指标的 CSV 路径")
    # Server-side ZO CLI (renamed for clarity)
    parser.add_argument("--server_zo_enable", action="store_true", help="Enable server-side ZO updates (clients report directional derivatives only)")
    parser.add_argument("--server_zo_dir_count", type=int, default=1, help="Number of ZO directions (seeds) per round on server")
    parser.add_argument("--server_zo_epsilon", type=float, default=1e-4, help="Epsilon used by clients when evaluating f(x±eps u) in server-side ZO")
    parser.add_argument("--server_zo_lr", type=float, default=1e-6, help="Server learning rate for applying ZO update")
    parser.add_argument("--server_zo_optimizer", type=str, default="sgd", choices=["sgd", "adam", "muon"], help="Server-side optimizer for ZO updates")
    parser.add_argument("--server_zo_weight_decay", type=float, default=0.0, help="Weight decay for server Adam/Muon")
    parser.add_argument("--server_zo_eps", type=float, default=1e-8, help="Epsilon for server Adam/Muon")
    parser.add_argument("--server_zo_betas", type=float, nargs=2, default=[0.9, 0.999], help="Betas for server Adam/Muon")
    parser.add_argument("--server_zo_muon_cautious", action="store_true", help="Enable Muon cautious mode on server")
    parser.add_argument("--server_zo_muon_orthogonal_init", action="store_true", help="Enable Muon orthogonal init on server (2D params)")
    parser.add_argument("--server_zo_muon_hidden_size", type=int, default=768, help="Muon hidden size on server for reshaping 1D params")
    args = parser.parse_args()

    # 设置可见 GPU（若指定）
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

    # 设备选择
    device, _, _ = resolve_runtime_device(args.device)

    # 向客户端下发本轮轮次/ZO 配置
    if args.instruct_enable:
        strategy = InstructFLStrategy(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.min_available_clients,
            candidate_pool=args.instruct_candidate_pool,
            topk=args.instruct_topk,
            dir_count=args.instruct_dir_count,
            epsilon=args.server_zo_epsilon,
            server_lr=args.server_zo_lr,
            device=device,
            optimizer_name=args.server_zo_optimizer,
            weight_decay=args.server_zo_weight_decay,
            eps=args.server_zo_eps,
            betas=tuple(args.server_zo_betas),
            muon_cautious=args.server_zo_muon_cautious,
            muon_orthogonal_init=args.server_zo_muon_orthogonal_init,
            muon_hidden_size=args.server_zo_muon_hidden_size,
            eval_steps=args.instruct_eval_steps,
            server_csv_path=args.instruct_server_csv,
        )
    elif args.server_zo_enable:
        strategy = ServerSideZOStrategy(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.min_available_clients,
            dir_count=args.server_zo_dir_count,
            epsilon=args.server_zo_epsilon,
            server_lr=args.server_zo_lr,
            device=device,
            optimizer_name=args.server_zo_optimizer,
            weight_decay=args.server_zo_weight_decay,
            eps=args.server_zo_eps,
            betas=tuple(args.server_zo_betas),
            muon_cautious=args.server_zo_muon_cautious,
            muon_orthogonal_init=args.server_zo_muon_orthogonal_init,
            muon_hidden_size=args.server_zo_muon_hidden_size,
        )
    else:
        def fit_config(rnd: int):
            return {"server_round": rnd}
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=args.min_available_clients,
            on_fit_config_fn=fit_config,
        )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

import argparse
from typing import Optional, Dict, Any, List, Tuple

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import base64

# 将项目根目录加入 sys.path，方便导入根目录下的模块（例如 optim_muon）
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.optimizer_name = str(optimizer_name).lower()
        self.weight_decay = float(weight_decay)
        self.eps = float(eps)
        self.betas = (float(betas[0]), float(betas[1]))
        self.muon_cautious = bool(muon_cautious)
        self.muon_orthogonal_init = bool(muon_orthogonal_init)
        self.muon_hidden_size = int(muon_hidden_size)
        self.eval_steps = int(eval_steps)
        # state
        self._selected_seeds: List[int] = []
        self._selected_dirs: List[List[np.ndarray]] = []
        self._opt = None  # type: ignore[var-annotated]
        self._params = None  # type: ignore[var-annotated]

    def _phase(self, rnd: int) -> str:
        # 交替执行：奇数轮 BP 选择，偶数轮 ZO 评估
        return "bp_select" if (rnd % 2 == 1) else "zo_eval"

    def _make_seeds_pool(self, rnd: int) -> List[int]:
        device = torch.device("cuda") if self.device == "cuda" else torch.device("cpu")
        try:
            gen = torch.Generator(device=device).manual_seed(20231111 + rnd)
        except Exception:
            gen = torch.Generator().manual_seed(20231111 + rnd)
        seeds = torch.randint(
            low=0,
            high=2**31 - 1,
            size=(max(1, self.candidate_pool),),
            generator=gen,
            device=device,
            dtype=torch.int64,
        )
        return [int(x.item()) for x in seeds.to("cpu")]

    def _on_fit_config_fn(self, rnd: int) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"server_round": rnd, "instruct_enable": True, "instruct_phase": self._phase(rnd)}
        if cfg["instruct_phase"] == "bp_select":
            cfg["bp_candidate_seeds_json"] = json.dumps(self._make_seeds_pool(rnd))
            cfg["bp_select_topk"] = int(self.topk)
            cfg["instruct_dir_count"] = int(self.dir_count)
            cfg["instruct_cosine_target"] = 0.9
            cfg["instruct_dir_method"] = "r"
        else:
            cfg["zo_server_side"] = True
            # 优先下发 strong 客户端提供的“指令方向”，否则回退到 seeds
            if self._selected_dirs:
                payload: Dict[str, Any] = {"dtype": "float32", "shapes": [], "dirs": []}
                payload["shapes"] = [list(arr.shape) for arr in self._selected_dirs[0]]
                for one in self._selected_dirs[: self.dir_count]:
                    per_param_b64: List[str] = []
                    for arr in one:
                        buf = arr.astype(np.float32).tobytes(order="C")
                        per_param_b64.append(base64.b64encode(buf).decode("ascii"))
                    payload["dirs"].append(per_param_b64)
                cfg["instruct_dir_blob_json"] = json.dumps(payload)
            else:
                seeds = self._selected_seeds[:self.dir_count]
                if len(seeds) < self.dir_count:
                    # 退化：若未选满，用新随机种子补齐
                    need = self.dir_count - len(seeds)
                    seeds += self._make_seeds_pool(rnd)[:need]
                cfg["zo_dir_seeds_json"] = json.dumps([int(s) for s in seeds[: self.dir_count]])
            cfg["zo_epsilon"] = float(self.epsilon)
            cfg["zo_eval_steps"] = int(self.eval_steps)
        return cfg

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:  # type: ignore[override]
        if len(results) == 0:
            return super().aggregate_fit(rnd, results, failures)

        phase = self._phase(rnd)
        # 使用首个返回值的参数作为基准透传（本轮不在服务端聚合参数）
        base_params = results[0][1].parameters
        if base_params is None:
            return super().aggregate_fit(rnd, results, failures)
        nds = parameters_to_ndarrays(base_params)
        torch_device = torch.device("cuda") if self.device == "cuda" else torch.device("cpu")
        torch_device_str = "cuda" if self.device == "cuda" else "cpu"

        # 汇总通信字节（若客户端上报）
        in_bytes_list: List[float] = []
        out_bytes_list: List[float] = []
        for _, res in results:
            m = res.metrics or {}
            in_bytes = float(m.get("comm_in_bytes", 0.0))
            out_bytes = float(m.get("comm_out_bytes", 0.0))
            if in_bytes > 0:
                in_bytes_list.append(in_bytes)
            if out_bytes > 0:
                out_bytes_list.append(out_bytes)

        if phase == "bp_select":
            # 统计 strong 客户端上报的种子并投票
            freq: Dict[int, int] = {}
            captured_dirs: List[List[np.ndarray]] = []
            selected_shapes: List[Tuple[int, ...]] = []
            for _, res in results:
                m = res.metrics or {}
                s_json = m.get("bp_top_seeds_json")
                if s_json:
                    try:
                        top_seeds = json.loads(s_json)
                        for s in top_seeds:
                            ss = int(s)
                            freq[ss] = freq.get(ss, 0) + 1
                    except Exception:
                        pass
                # 解析 strong 客户端生成的“指令方向”
                blob_json = m.get("instruct_dir_blob_json")
                if blob_json:
                    try:
                        blob = json.loads(blob_json)
                        shapes = [tuple(x) for x in blob.get("shapes", [])]
                        dirs_b64 = blob.get("dirs", [])
                        if shapes and dirs_b64:
                            if not selected_shapes:
                                selected_shapes = list(shapes)
                            if list(shapes) == selected_shapes:
                                for one in dirs_b64:
                                    per_param: List[np.ndarray] = []
                                    for shp, b64 in zip(shapes, one):
                                        raw = base64.b64decode(b64.encode("ascii"))
                                        arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(shp)
                                        per_param.append(arr)
                                    captured_dirs.append(per_param)
                    except Exception:
                        pass
            # 选择出现次数最多的 dir_count 个
            ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
            chosen = [int(s) for (s, _) in ranked[: self.dir_count]]
            if len(chosen) < self.dir_count:
                # 补齐
                pool = self._make_seeds_pool(rnd)
                for s in pool:
                    if len(chosen) >= self.dir_count:
                        break
                    if int(s) not in chosen:
                        chosen.append(int(s))
            self._selected_seeds = chosen[: self.dir_count]
            # 保存 strong 客户端上报的“指令方向”（优先）
            if captured_dirs:
                self._selected_dirs = captured_dirs[: self.dir_count]
            else:
                self._selected_dirs = []

            avg_c2s = float(torch.tensor(out_bytes_list, device=torch_device, dtype=torch.float64).mean().item()) if out_bytes_list else 0.0
            avg_s2c = float(torch.tensor(in_bytes_list, device=torch_device, dtype=torch.float64).mean().item()) if in_bytes_list else 0.0
            metrics: Dict[str, fl.common.Scalar] = {
                "instruct_phase": "bp_select",
                "selected_count": len(self._selected_seeds),
                "selected_dirs_count": len(self._selected_dirs),
                "avg_comm_client_to_server_bytes": avg_c2s,
                "avg_comm_server_to_client_bytes": avg_s2c,
            }
            try:
                print(f"[Instruct][BP-Select][Round {rnd}] selected={len(self._selected_seeds)} dirs={len(self._selected_dirs)} "
                      f"avg_c->s={metrics['avg_comm_client_to_server_bytes']:.1f}B "
                      f"avg_s->c={metrics['avg_comm_server_to_client_bytes']:.1f}B "
                      f"seeds={self._selected_seeds}", flush=True)
            except Exception:
                pass
            # 直接透传参数（不聚合）
            return ndarrays_to_parameters(nds), metrics

        # zo_eval 阶段：重建方向并应用优化器更新
        w_tensors = [torch.from_numpy(arr).to(device=torch_device, dtype=torch.float32) for arr in nds]
        grad_tensors = [torch.zeros_like(w, device=torch_device) for w in w_tensors]
        # 若 strong 客户端已提供方向，则使用其构造梯度；否则回退到 seeds
        seeds: List[int] = []
        use_dirs = bool(self._selected_dirs)
        dir_len = len(self._selected_dirs) if use_dirs else len(self._selected_seeds)
        if not use_dirs:
            seeds = [int(s) for s in (self._selected_seeds[: self.dir_count] if self._selected_seeds else [])]
            if len(seeds) == 0:
                seeds = self._make_seeds_pool(rnd)[: self.dir_count]
            dir_len = len(seeds)

        dir_sums = torch.zeros(dir_len, device=torch_device, dtype=torch.float64)
        dir_weights = torch.zeros(dir_len, device=torch_device, dtype=torch.float64)
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
            trimmed = g_list[:dir_len]
            if len(trimmed) == 0:
                continue
            g_tensor = torch.tensor(trimmed, device=torch_device, dtype=torch.float64)
            effective_len = g_tensor.numel()
            dir_sums[:effective_len] += g_tensor * num_ex
            dir_weights[:effective_len] += num_ex
        if dir_len > 0:
            dir_means_tensor = torch.zeros(dir_len, device=torch_device, dtype=torch.float64)
            valid_mask = dir_weights > 0
            if bool(valid_mask.any().item()):
                dir_means_tensor[valid_mask] = dir_sums[valid_mask] / dir_weights[valid_mask]
            dir_means = dir_means_tensor.tolist()
        else:
            dir_means = []

        dir_cnt = max(1, dir_len)
        if use_dirs:
            for i, g_bar in enumerate(dir_means):
                one = self._selected_dirs[i]
                for idx, arr in enumerate(one):
                    u = torch.from_numpy(arr.astype(np.float32)).to(device=torch_device)
                    grad_tensors[idx] = grad_tensors[idx] + (float(g_bar) * u) / dir_cnt
        else:
            for seed, g_bar in zip(seeds, dir_means):
                gen = torch.Generator(device=torch_device_str).manual_seed(int(seed))
                for idx, w in enumerate(w_tensors):
                    u = torch.randn(w.shape, generator=gen, device=w.device, dtype=w.dtype)
                    grad_tensors[idx] = grad_tensors[idx] + (float(g_bar) * u) / dir_cnt

        opt_name = self.optimizer_name
        if opt_name == "sgd":
            new_w_tensors = [ w - self.server_lr * g for (w, g) in zip(w_tensors, grad_tensors) ]
            new_nds = [ t.detach().to("cpu").numpy() for t in new_w_tensors ]
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
            for p, g in zip(self._params, grad_tensors):
                p.grad = g.clone().to(p.device)
            self._opt.step()
            new_nds = [ p.data.detach().to("cpu").numpy() for p in self._params ]

        avg_c2s = float(torch.tensor(out_bytes_list, device=torch_device, dtype=torch.float64).mean().item()) if out_bytes_list else 0.0
        avg_s2c = float(torch.tensor(in_bytes_list, device=torch_device, dtype=torch.float64).mean().item()) if in_bytes_list else 0.0
        metrics: Dict[str, fl.common.Scalar] = {
            "instruct_phase": "zo_eval",
            "dir_count": dir_len,
            "zo_optimizer": self.optimizer_name,
            "zo_lr": float(self.server_lr),
            "avg_comm_client_to_server_bytes": avg_c2s,
            "avg_comm_server_to_client_bytes": avg_s2c,
        }
        try:
            print(f"[Instruct][ZO-Eval][Round {rnd}] dirs={dir_len} "
                  f"avg_c->s={metrics['avg_comm_client_to_server_bytes']:.1f}B "
                  f"avg_s->c={metrics['avg_comm_server_to_client_bytes']:.1f}B", flush=True)
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
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
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

    def _make_dir_seeds(self, rnd: int) -> List[int]:
        device = torch.device("cuda") if self.device == "cuda" else torch.device("cpu")
        try:
            gen = torch.Generator(device=device).manual_seed(12345 + rnd)
        except Exception:
            gen = torch.Generator().manual_seed(12345 + rnd)
        seeds = torch.randint(
            low=0,
            high=2**31 - 1,
            size=(self.dir_count,),
            generator=gen,
            device=device,
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
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:  # type: ignore[override]
        # 若无结果或失败，回退到父类逻辑
        if len(results) == 0:
            return super().aggregate_fit(rnd, results, failures)

        # 从第一个返回中获取当前参数形状
        first_params = results[0][1].parameters
        if first_params is None:
            return super().aggregate_fit(rnd, results, failures)
        nds = parameters_to_ndarrays(first_params)
        param_shapes = [arr.shape for arr in nds]
        torch_device = torch.device("cuda") if self.device == "cuda" else torch.device("cpu")
        torch_device_str = "cuda" if self.device == "cuda" else "cpu"

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
        w_tensors = [torch.from_numpy(arr).to(device=torch_device, dtype=torch.float32) for arr in nds]
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
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

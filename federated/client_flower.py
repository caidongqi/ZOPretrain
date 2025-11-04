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

import flwr as fl

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

        # Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = create_model(len(self.tokenizer)).to(self.device)
        self.model.train()

        # Trainable parameters subset
        self.trainable_params: List[torch.nn.Parameter] = get_trainable_parameters(self.model, self.scope)
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

        start_time = time.time()
        total_loss = 0.0
        step_count = 0
        server_round = int(config.get('server_round', self.round_idx + 1))

        # Server-side ZO path: clients do NOT update params; only report directional derivatives
        if (self.mode == 'ZO') and bool(config.get('zo_server_side', False)):
            epsilon = float(config.get('zo_epsilon', 1e-4))
            dir_seeds = list(config.get('zo_dir_seeds', []))
            eval_steps = int(config.get('zo_eval_steps', 1))

            def _loss_with_perturb(sign: float, dirs: List[torch.Tensor]) -> float:
                # apply
                for p, u in zip(self.trainable_params, dirs):
                    p.data.add_(sign * epsilon * u.to(p.device))
                try:
                    self.model.eval()
                    total = 0.0
                    steps = 0
                    with torch.no_grad():
                        for batch in self.trainloader:
                            inputs = batch.to(self.device)
                            labels = inputs.clone()
                            logits = self.model(inputs).logits
                            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                            total += float(loss.item())
                            steps += 1
                            if steps >= eval_steps:
                                break
                    return total / max(steps, 1)
                finally:
                    # revert
                    for p, u in zip(self.trainable_params, dirs):
                        p.data.add_(-sign * epsilon * u.to(p.device))
                    self.model.train()

            # build param-shape-aligned directions per seed
            g_dir_list: List[float] = []
            for seed in dir_seeds:
                gen = torch.Generator(device=self.device if self.device == 'cuda' else 'cpu').manual_seed(int(seed))
                dirs: List[torch.Tensor] = []
                for p in self.trainable_params:
                    u = torch.randn_like(p.data, generator=gen)
                    dirs.append(u)
                lp = _loss_with_perturb(+1.0, dirs)
                lm = _loss_with_perturb(-1.0, dirs)
                g_dir = (lp - lm) / (2.0 * epsilon)
                g_dir_list.append(float(g_dir))

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
                    logits = self.model(inputs).logits
                    loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss.backward()
                    # 计算梯度范数（FO）
                    grad_norm_sq = 0.0
                    for p in self.trainable_params:
                        if p.grad is not None:
                            g = p.grad.detach()
                            grad_norm_sq += float(torch.sum(g * g).item())
                    grad_norm = float(grad_norm_sq) ** 0.5
                    self.optimizer.step()
                else:  # ZO
                    with torch.no_grad():
                        logits = self.model(inputs).logits
                        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

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
                    grad_norm_sq = 0.0
                    for g in grad_paramwise:
                        if g is not None:
                            gg = g.detach()
                            grad_norm_sq += float(torch.sum(gg * gg).item())
                    grad_norm = float(grad_norm_sq) ** 0.5

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

                total_loss += float(loss.item())
                step_count += 1

                # 记录 CSV（按步）
                if self.csv_file is not None and (step_count % self.log_interval == 0):
                    with open(self.csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            server_round,
                            epoch,
                            step_count,
                            self.mode,
                            self.scope,
                            self.q if self.mode == 'ZO' else 'N/A',
                            self.lr,
                            self.batch_size,
                            float(loss.item()),
                            grad_norm,
                            self.client_id,
                        ])
                if self.local_steps is not None and step_count >= self.local_steps:
                    break
            if self.local_steps is not None and step_count >= self.local_steps:
                break

        metrics = {
            "client_id": self.client_id,
            "train_time_s": time.time() - start_time,
            "avg_loss": total_loss / max(step_count, 1),
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
        total_loss = 0.0
        steps = 0
        with torch.no_grad():
            for batch in self.trainloader:
                inputs = batch.to(self.device)
                labels = inputs.clone()
                logits = self.model(inputs).logits
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += float(loss.item())
                steps += 1
                if steps >= 20:  # limit evaluation cost
                    break
        self.model.train()
        return total_loss / max(steps, 1), len(self.trainloader.dataset), {"eval_steps": steps}


def main():
    parser = argparse.ArgumentParser(description="Flower client for ZO/FO training")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--mode", type=str, choices=["FO", "ZO"], default="ZO")
    parser.add_argument("--scope", type=str, choices=["full", "reduced"], default="full")
    parser.add_argument("--q", type=int, default=1)
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
    parser.add_argument("--gpu", type=int, default=4, help="当使用 CUDA 时选择 GPU 序号，如 0、1。")
    parser.add_argument("--csv_file", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()

    # 在任何 CUDA 检查/使用前设置可见 GPU（若指定）
    if (args.device in ("auto", "cuda")) and (args.gpu is not None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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
    if args.gpu is not None:
        ops_parts.append(f"gpu{args.gpu}")
    ops = "_".join(ops_parts)

    exp_dir = Path("results") / (
        f"{args.mode}_{args.optimizer}_{args.scope}_n{args.num_clients}_q{args.q}_"
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
                "q": args.q,
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
                "csv_file": auto_csv,
                "log_interval": args.log_interval,
                "exp_dir": str(exp_dir),
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    client = ZOFLClient(
        client_id=args.client_id,
        num_clients=args.num_clients,
        mode=args.mode,
        scope=args.scope,
        q=args.q,
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



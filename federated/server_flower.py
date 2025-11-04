import argparse
from typing import Optional, Dict, Any, List, Tuple

import json
import numpy as np
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


class ServerSideZOStrategy(fl.server.strategy.FedAvg):
    """Server-side ZO updater using seeds and client-computed directional derivatives.

    - Sends seeds and epsilon to clients via on_fit_config_fn
    - Expects clients to return metrics with key 'zo_dir_g_json' containing a JSON list of
      directional derivatives (one per seed): g^T u_i ≈ (f(x+e u_i) - f(x-e u_i))/(2e)
    - Reconstructs directions with torch.randn using the same seeds and current parameter shapes
    - Forms gradient estimate: sum_i (bar_g_i * u_i) / dir_count, optionally weighted by num_examples
    - Updates global parameters with server_lr (simple SGD)
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
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=min_available_clients,
        )
        self.dir_count = int(dir_count)
        self.epsilon = float(epsilon)
        self.server_lr = float(server_lr)

    def _make_dir_seeds(self, rnd: int) -> List[int]:
        rng = np.random.default_rng(seed=12345 + rnd)
        return [int(x) for x in rng.integers(low=0, high=2**31-1, size=self.dir_count)]

    def on_fit_config_fn(self, rnd: int) -> Dict[str, Any]:  # type: ignore[override]
        cfg = {"server_round": rnd}
        # 指示客户端启用 server-side ZO 汇报
        cfg["zo_server_side"] = True
        cfg["zo_dir_seeds"] = self._make_dir_seeds(rnd)
        cfg["zo_epsilon"] = self.epsilon
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

        # 聚合每个方向的标量导数（按样本数加权）
        seed_list = self._make_dir_seeds(rnd)
        dir_sums = np.zeros(len(seed_list), dtype=np.float64)
        dir_weights = np.zeros(len(seed_list), dtype=np.float64)
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
            for i, val in enumerate(g_list[:len(seed_list)]):
                dir_sums[i] += float(val) * num_ex
                dir_weights[i] += num_ex
        # 计算每个方向的平均导数
        dir_means = [ (dir_sums[i] / dir_weights[i]) if dir_weights[i] > 0 else 0.0 for i in range(len(seed_list)) ]

        # 重建方向并形成全量梯度估计
        grad_nds = [np.zeros_like(arr, dtype=np.float32) for arr in nds]
        for seed, g_bar in zip(seed_list, dir_means):
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            for idx, shape in enumerate(param_shapes):
                u = torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32).numpy()
                grad_nds[idx] += (g_bar * u) / max(1, len(seed_list))

        # 执行一次 SGD 更新
        new_nds = [ w - self.server_lr * g for (w, g) in zip(nds, grad_nds) ]
        new_params = ndarrays_to_parameters(new_nds)

        metrics: Dict[str, fl.common.Scalar] = {
            "server_side_zo": True,
            "dir_count": len(seed_list),
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
    parser.add_argument("--zo_server_side", action="store_true", help="Enable server-side ZO updates")
    parser.add_argument("--zo_dir_count", type=int, default=1, help="Number of ZO directions per round")
    parser.add_argument("--zo_epsilon", type=float, default=1e-4, help="ZO epsilon for finite differences")
    parser.add_argument("--zo_server_lr", type=float, default=1e-6, help="Server learning rate for ZO update")
    args = parser.parse_args()

    # 向客户端下发本轮轮次/ZO 配置
    if args.zo_server_side:
        strategy = ServerSideZOStrategy(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.min_available_clients,
            dir_count=args.zo_dir_count,
            epsilon=args.zo_epsilon,
            server_lr=args.zo_server_lr,
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

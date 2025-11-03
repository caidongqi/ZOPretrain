import argparse
from typing import Optional, Dict, Any

import flwr as fl


def main():
    parser = argparse.ArgumentParser(description="Flower server for ZO/FO federated learning")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min_fit_clients", type=int, default=2)
    parser.add_argument("--min_available_clients", type=int, default=2)
    parser.add_argument("--fraction_fit", type=float, default=1.0)
    parser.add_argument("--fraction_evaluate", type=float, default=0.0)
    args = parser.parse_args()

    # 向客户端下发本轮轮次，便于客户端 CSV 精确记录 server_round
    def fit_config(rnd: int):
        return {"server_round": rnd}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=0,
        min_available_clients=args.min_available_clients,
        on_fit_config_fn=fit_config,
        # We aggregate only the list of arrays provided by clients (trainable subset)
        # so default parameters/aggregation works well.
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()



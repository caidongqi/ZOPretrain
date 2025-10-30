import math
import warnings
from typing import Callable, Tuple, Optional

import torch
from torch.optim import Optimizer
import copy


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero.
    See provided reference implementation from transformers/optimization.py derived Muon.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class AdamW(Optimizer):
    """
    Muon-style AdamW variant with optional cautious mode and orthogonal init.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
        cautious: bool = False,
        orthogonal_init: bool = False,
        muon_exclude: Optional[dict] = None,
        hidden_size: int = 768,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version.",
                FutureWarning,
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }

        self.hidden_size = hidden_size
        self.cautious = cautious
        self.muon_exclude = muon_exclude or {}

        if orthogonal_init:
            orthogonal_params = copy.deepcopy(params)
            for each_group in orthogonal_params:
                for each in each_group["params"]:
                    if each.ndim == 2:
                        each.data = zeropower_via_newtonschulz5(each, 5).to(each.dtype)
            super().__init__(orthogonal_params, defaults)
        else:
            super().__init__(params, defaults)

    def adjust_lr_for_muon(self, lr: float, param_shape: Tuple[int, int]) -> float:
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decoupled weight decay
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # EMA updates
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute normalized gradient
                if self.cautious:
                    mask = (exp_avg * grad > 0).to(grad.dtype)
                    mask = mask * (mask.numel() / (mask.sum() + 1))
                    norm_grad = (exp_avg * mask) / denom
                else:
                    norm_grad = exp_avg / denom

                # reshape 1D momentum to 2D for Muon orthogonalization when applicable
                if norm_grad.ndim == 1:
                    hs = self.hidden_size
                    if len(norm_grad) % hs == 0 and len(norm_grad) > hs:
                        norm_grad = norm_grad.reshape(hs, -1)
                    if norm_grad.ndim == 2:
                        step_size = self.adjust_lr_for_muon(step_size, norm_grad.shape)
                        norm_grad = zeropower_via_newtonschulz5(norm_grad, 5).to(norm_grad.dtype)
                        norm_grad = norm_grad.view(-1)

                p.add_(norm_grad, alpha=-step_size)

        return loss


class SGD(Optimizer):
    def __init__(self, params, defaults) -> None:
        super().__init__(params, defaults)



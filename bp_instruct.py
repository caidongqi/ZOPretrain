import argparse
import csv
import json
import logging
import math
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

# # 导入自定义模块
# from model import create_model
# from data import get_dataloader

# --- 常量配置 (Constants) ---

INSTRUCT_COSINE_TARGET = 0.9

# --- 1. 配置与模型定义 (Configuration and Model Definition) ---
# 注意: create_model 函数现在从 model.py 导入

def get_trainable_parameters(model, scope='full'):
    """
    根据指定的范围，返回可训练的参数列表。
    - 'full': 返回所有参数。
    - 'reduced': 只返回最后一层 (MLP + LayerNorm) 的参数。
    """
    if scope == 'full':
        print("Training scope: full model.")
        return list(model.parameters())
    
    elif scope == 'reduced':
        print("Training scope: reduced (only the last transformer block's MLP and LayerNorm).")
        trainable_params = []
        # 选择最后一层进行训练，以复现论文的核心发现
        last_block = model.transformer.h[-1]
        for name, param in last_block.named_parameters():
            if 'mlp' in name or 'ln_2' in name:
                trainable_params.append(param)
    
        # 同时，输出层也需要训练
        for param in model.lm_head.parameters():
            trainable_params.append(param)
            
        return trainable_params
    else:
        raise ValueError(f"Unknown training scope: {scope}")

# --- 2. 数据加载与预处理 (Data Loading and Preprocessing) ---
# 注意: get_dataloader 函数现在从 data.py 导入


# --- 3. 核心算法：ZO 梯度估计 (Core Algorithm: ZO Gradient Estimator) ---

@torch.no_grad()
def zo_gradient_estimator(
    model,
    trainable_params,
    loss_fn,
    inputs,
    labels,
    q,
    epsilon,
    device,
    manual_directions=None,
    data_provider=None,
):
    """ZO梯度估计器，支持可迭代的手动方向序列，可选地为每个查询提供独立数据。"""
    # 关闭dropout，加速且去噪
    was_training = model.training
    model.eval()

    # 记录可训练参数的原值
    original = [p.data.clone() for p in trainable_params]

    def compute_loss(batch_inputs, batch_labels):
        logits = model(batch_inputs).logits
        # Shift logits and labels for next-token prediction
        # logits[..., :-1, :] predicts labels[..., 1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_labels[..., 1:].contiguous()
        return loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    def get_batch():
        if data_provider is None:
            return inputs, labels
        batch_inputs, batch_labels = data_provider()
        if batch_inputs.device != device:
            batch_inputs = batch_inputs.to(device)
        if batch_labels.device != device:
            batch_labels = batch_labels.to(device)
        return batch_inputs, batch_labels

    grads = [torch.zeros_like(p.data) for p in trainable_params]
    used_directions = 0
    manual_used = 0

    manual_iter = None
    if manual_directions is not None:
        manual_iter = iter(manual_directions)

    if manual_iter is not None:
        while True:
            if q is not None and manual_used >= q:
                break
            try:
                raw_direction = next(manual_iter)
            except StopIteration:
                break
            if raw_direction is None:
                continue

            direction = []
            for p, d in zip(trainable_params, raw_direction):
                dt = d.detach()
                if dt.device != device or dt.dtype != p.data.dtype:
                    dt = dt.to(device=device, dtype=p.data.dtype)
                direction.append(dt)

            batch_inputs, batch_labels = get_batch()

            for p, p0, d in zip(trainable_params, original, direction):
                p.data = p0 + epsilon * d
            loss_pos = compute_loss(batch_inputs, batch_labels)

            for p, p0, d in zip(trainable_params, original, direction):
                p.data = p0 - epsilon * d
            loss_neg = compute_loss(batch_inputs, batch_labels)

            for p, p0 in zip(trainable_params, original):
                p.data = p0.clone()

            proj = (loss_pos - loss_neg) / (2 * epsilon)
            for gi, d in enumerate(direction):
                grads[gi].add_(proj * d)

            used_directions += 1
            manual_used += 1

    remaining_q = 0
    if q is not None:
        remaining_q = max(q - manual_used, 0)

    # 随机方向部分，继续使用种子以节省显存
    seeds = []
    proj_grads = []
    for _ in range(remaining_q):
        seed = torch.randint(0, 2**31 - 1, ()).item()
        seeds.append(seed)

        batch_inputs, batch_labels = get_batch()

        torch.manual_seed(seed)
        for p in trainable_params:
            z = torch.randn_like(p.data)
            p.data = p.data + epsilon * z
        loss_pos = compute_loss(batch_inputs, batch_labels)

        torch.manual_seed(seed)
        for p, p0 in zip(trainable_params, original):
            z = torch.randn_like(p.data)
            p.data = p0 - epsilon * z
        loss_neg = compute_loss(batch_inputs, batch_labels)

        for p, p0 in zip(trainable_params, original):
            p.data = p0.clone()

        proj_grads.append(((loss_pos - loss_neg) / (2 * epsilon)).item())

    # 重建随机方向贡献
    for seed, proj in zip(seeds, proj_grads):
        torch.manual_seed(seed)
        for gi, p in enumerate(trainable_params):
            z = torch.randn_like(p.data)
            grads[gi].add_(proj * z)
        used_directions += 1

    if used_directions > 0:
        for gi in range(len(grads)):
            grads[gi].div_(float(used_directions))

    if was_training:
        model.train()

    return grads


def save_latest_checkpoint(
    model,
    tokenizer,
    checkpoint_dir,
    optimizer_state=None,
    metadata=None,
    logger=None,
):
    if not checkpoint_dir:
        return

    ckpt_path = Path(checkpoint_dir)
    if ckpt_path.exists():
        shutil.rmtree(ckpt_path)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_path / "tokenizer")

    if optimizer_state is not None:
        torch.save(optimizer_state, ckpt_path / "optimizer.pt")

    if metadata is not None:
        with open(ckpt_path / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    if logger is not None:
        logger.info("Checkpoint saved to %s", ckpt_path)


def compute_backprop_gradients(model, trainable_params, loss_fn, inputs, labels):
    """执行一次标准BP，返回loss和每个参数的梯度副本。"""
    for p in trainable_params:
        if p.grad is not None:
            p.grad.zero_()

    with torch.enable_grad():
        logits = model(inputs).logits
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    loss.backward()

    grads = []
    for p in trainable_params:
        if p.grad is None:
            grads.append(torch.zeros_like(p.data))
        else:
            grads.append(p.grad.detach().clone())

    model.zero_grad(set_to_none=False)

    return loss.detach(), grads


def generate_instruct_directions_with_R(
    bp_grads,
    q,
    cosine_target,
    total_norm,
    total_norm_sq=None,
    device=None,
    *,
    min_cos_floor=None,
    seed=None,
    max_rank=None,
    tune_to_target=False,
    use_float32=True,
):
    """
    基于“能量覆盖率”的稀疏方向生成，使其与 BP 梯度具有较高余弦相似度。
    改进点：
    - 可选的最小相似度下限 `min_cos_floor`（默认不强制 0.9 下限）
    - 可重复性：`seed` 控制候选随机采样
    - Rank 限制：`max_rank` 作为显式上限
    - 数值稳定：flatten/统计阶段可统一为 float32
    - 可选“贴近目标”微调：若达到的相似度 > 目标且 `tune_to_target=True`，加入少量正交噪声将余弦拉向目标值
    """
    if not bp_grads or q is None or q <= 0:
        return None
    if total_norm <= 1e-12:
        return None

    if device is None:
        device = bp_grads[0].device if hasattr(bp_grads[0], 'device') else torch.device('cpu')

    cosine_target = float(cosine_target)
    cosine_target = max(min(cosine_target, 0.9999), -0.9999)
    target_sign = -1.0 if cosine_target < 0.0 else 1.0

    # 最小相似度阈值：默认不强加 0.9 下限；若用户提供 min_cos_floor，则取 max(|c|, min_cos_floor)
    min_similarity = abs(cosine_target)
    if min_cos_floor is not None:
        min_similarity = max(min_similarity, float(min_cos_floor))
    min_similarity = min(min_similarity, 0.9999)
    eps = 1e-12

    # 统计阶段用 float32 提升稳定性
    def _to_stat_dtype(t):
        if use_float32:
            return t.to(device=device, dtype=torch.float32)
        return t.to(device=device)

    grad_flat = torch.cat([_to_stat_dtype(g).flatten() for g in bp_grads])
    d = grad_flat.numel()
    # 允许复用外部预计算的 total_norm_sq
    if total_norm_sq is None:
        total_norm_sq = float(torch.sum(grad_flat * grad_flat).item())
    if total_norm_sq <= eps:
        return None

    energy_threshold = min_similarity ** 2 * total_norm_sq

    abs_sq = grad_flat.abs().pow(2)
    # rank 限制：初始 k 不超过 max_rank（若提供）
    initial_rank = min(64, d)
    if max_rank is not None:
        initial_rank = min(initial_rank, int(max_rank))
    k = max(1, initial_rank)

    while True:
        values, indices = torch.topk(abs_sq, k, largest=True)
        captured_energy = float(values.sum().item())
        # Rank 上限
        k_limit = d if max_rank is None else min(d, int(max_rank))
        if captured_energy >= energy_threshold or k >= k_limit:
            break
        previous_k = k
        k = min(k * 2, k_limit)
        if k == previous_k:
            break

    cumsum_values = torch.cumsum(values, dim=0)
    if energy_threshold <= float(cumsum_values[-1].item()):
        effective_rank_idx = torch.searchsorted(cumsum_values, torch.tensor(energy_threshold, device=cumsum_values.device))
        effective_rank = int(effective_rank_idx.item()) + 1
    else:
        effective_rank = int(values.numel())

    # 最终 rank 受 max_rank 约束
    effective_rank = max(1, min(effective_rank, values.numel()))
    if max_rank is not None:
        effective_rank = min(effective_rank, int(max_rank))
    # 候选索引池：使用更大的集合以支持随机采样多样性
    candidate_pool_size = min(max(effective_rank * 2, 128), k)
    candidate_indices = indices[:candidate_pool_size]

    # 随机生成器（可重复）
    gen = None
    if seed is not None:
        try:
            gen = torch.Generator(device=device).manual_seed(int(seed))
        except Exception:
            gen = None

    def generator():
        total_start_time = time.time()
        total_cosine_sim = 0.0
        total_rank = 0
        
        for idx in range(q):
            iter_start_time = time.time()

            # 每次随机选择不同的索引集合
            # 从候选池中随机采样，确保能量满足阈值
            max_selected = min(candidate_pool_size, effective_rank * 2)
            
            # 尝试不同的采样策略：随机选择索引
            num_selected = effective_rank
            for attempt in range(10):  # 最多尝试10次
                # 随机打乱候选索引并选择前 num_selected 个（支持带 generator 的可重复性）
                try:
                    perm = torch.randperm(candidate_pool_size, device=device, generator=gen)
                except TypeError:
                    # 部分版本不支持 randperm 的 generator 参数，退化方案
                    rnd = torch.rand(candidate_pool_size, device=device)
                    perm = torch.argsort(rnd)
                selected_from_pool = perm[:num_selected]
                selected_indices = candidate_indices[selected_from_pool]
                
                # 检查能量是否满足阈值
                selected_abs_sq = abs_sq[selected_indices]
                captured_energy = float(selected_abs_sq.sum().item())
                
                if captured_energy >= energy_threshold:
                    break
                # 如果能量不够，增加选择的索引数量
                num_selected = min(num_selected + max(1, effective_rank // 4), max_selected)
            
            # 如果还是不够，使用top-k策略作为fallback
            if captured_energy < energy_threshold:
                selected_indices = indices[:effective_rank]

            # 基于选中的索引生成方向
            projection_flat = torch.zeros_like(grad_flat)
            projection_flat[selected_indices] = grad_flat[selected_indices]

            proj_norm = projection_flat.norm() + eps
            if proj_norm <= eps:
                projection_flat = grad_flat.clone()
            else:
                projection_flat = projection_flat * (total_norm / proj_norm)
            projection_flat = projection_flat * target_sign

            direction_flat = projection_flat
            direction_norm = direction_flat.norm() + eps
            actual_cosine_sim = torch.dot(direction_flat, grad_flat) / (direction_norm * total_norm)

            # 可选微调：若达到的相似度高于目标，希望“更贴近”目标值，则添加正交噪声降低余弦
            # d' = p + alpha * n，n ⟂ g, ||n|| = ||g|| => cos(d', g) = cos(p,g)/sqrt(1+alpha^2)
            # 令目标 c_t = |cosine_target|，若 actual_cosine_sim > c_t，则 alpha = sqrt((cos/c_t)^2 - 1)
            if tune_to_target:
                ct = abs(float(cosine_target))
                ct = min(max(ct, 0.0), 0.9999)
                cos_now = float(actual_cosine_sim.item())
                if cos_now > ct + 1e-6:
                    # 生成与 g 正交的单位噪声
                    # n = (z - proj_z) / ||...||，其中 proj_z = (z·g / ||g||^2) g
                    try:
                        z = torch.randn_like(direction_flat, generator=gen)
                    except Exception:
                        z = torch.randn_like(direction_flat)
                    proj_coeff = float(torch.dot(z, grad_flat).item()) / (total_norm_sq + eps)
                    n = z - proj_coeff * grad_flat
                    n_norm = float(n.norm().item())
                    if n_norm > eps:
                        n = n / (n_norm + eps)
                        # alpha 计算
                        alpha = math.sqrt(max((cos_now / (ct + eps))**2 - 1.0, 0.0))
                        direction_flat = direction_flat + (alpha * total_norm) * n
                        # 重新计算实际余弦
                        direction_norm = direction_flat.norm() + eps
                        actual_cosine_sim = torch.dot(direction_flat, grad_flat) / (direction_norm * total_norm)

            # 累积统计信息
            total_cosine_sim += actual_cosine_sim.item()
            total_rank += len(selected_indices)

            direction = []
            start = 0
            for g in bp_grads:
                sz = g.numel()
                out = direction_flat[start:start+sz].view_as(g)
                # 回写到与原梯度相同的 dtype / device
                if use_float32 and out.dtype != g.dtype:
                    out = out.to(dtype=g.dtype)
                if out.device != g.device:
                    out = out.to(g.device)
                direction.append(out)
                start += sz

            # 在最后一次迭代时打印汇总（放在 yield 之前，避免消费者停止后汇总丢失）
            if idx == q - 1:
                total_elapsed = time.time() - total_start_time
                avg_cosine_sim = total_cosine_sim / q
                avg_rank = total_rank / q
                print(
                    f"[Instruct-R] Summary: avg_rank={avg_rank:.1f}, avg_cos={avg_cosine_sim:.6f}, "
                    f"min_req={min_similarity:.4f}, tune={tune_to_target} total_time={total_elapsed:.4f}s",
                    flush=True,
                )

            yield direction

    return generator()

def generate_instruct_directions_blocked(
    bp_grads,
    q,
    cosine_target,
    total_norm,
    device=None,
    block_size=16384,
    rank_per_block=8,
    use_half_noise=True,
):
    """分块设置下基于低秩R投影生成高相似度的BP方向。"""
    del use_half_noise

    if not bp_grads or q <= 0:
        return None

    if device is None:
        device = bp_grads[0].device if hasattr(bp_grads[0], 'device') else torch.device('cpu')

    eps = 1e-12
    cosine_target = max(min(float(cosine_target), 0.9999), -0.9999)
    target_sign = -1.0 if cosine_target < 0.0 else 1.0

    min_similarity = max(abs(cosine_target), 0.9)
    min_similarity = min(min_similarity, 0.9999)

    grad_flat = torch.cat([g.flatten().to(device) for g in bp_grads])
    d = int(grad_flat.numel())
    if d == 0:
        return None

    total_norm_sq = float(torch.sum(grad_flat * grad_flat).item())
    if total_norm_sq <= eps:
        return None

    energy_threshold = min_similarity ** 2 * total_norm_sq

    abs_sq = grad_flat.abs().pow(2)
    block_size = max(int(block_size), 1)
    num_blocks = max(1, math.ceil(d / block_size))
    max_rank_cap = min(max(rank_per_block * num_blocks, 1), d)
    initial_rank = min(max(rank_per_block, 1), d)
    k = max(1, initial_rank)

    while True:
        values, indices = torch.topk(abs_sq, k, largest=True)
        captured_energy = float(values.sum().item())
        if captured_energy >= energy_threshold or k >= max_rank_cap:
            break
        next_k = min(k * 2, max_rank_cap)
        if next_k == k:
            break
        k = next_k

    if energy_threshold <= float(values.sum().item()):
        cumsum_values = torch.cumsum(values, dim=0)
        searched = torch.searchsorted(cumsum_values, torch.tensor(energy_threshold, device=cumsum_values.device))
        effective_rank = int(searched.item()) + 1
    else:
        effective_rank = int(values.numel())

    effective_rank = max(1, min(effective_rank, values.numel()))
    # 候选索引池：使用更大的集合以支持随机采样多样性
    candidate_pool_size = min(max(effective_rank * 2, 128), k)
    candidate_indices = indices[:candidate_pool_size]

    def generator():
        total_start_time = time.time()
        total_cosine_sim = 0.0
        total_rank = 0
        
        for idx in range(q):
            iter_start_time = time.time()

            # 每次随机选择不同的索引集合
            # 从候选池中随机采样，确保能量满足阈值
            max_selected = min(candidate_pool_size, effective_rank * 2)
            
            # 尝试不同的采样策略：随机选择索引
            num_selected = effective_rank
            captured_energy = 0.0
            for attempt in range(10):  # 最多尝试10次
                # 随机打乱候选索引并选择前num_selected个
                perm = torch.randperm(candidate_pool_size, device=device)
                selected_from_pool = perm[:num_selected]
                selected_indices = candidate_indices[selected_from_pool]
                
                # 检查能量是否满足阈值
                selected_abs_sq = abs_sq[selected_indices]
                captured_energy = float(selected_abs_sq.sum().item())
                
                if captured_energy >= energy_threshold:
                    break
                # 如果能量不够，增加选择的索引数量
                num_selected = min(num_selected + max(1, effective_rank // 4), max_selected)
            
            # 如果还是不够，使用top-k策略作为fallback
            if captured_energy < energy_threshold:
                selected_indices = indices[:effective_rank]

            # 基于选中的索引生成方向
            projection_flat = torch.zeros_like(grad_flat)
            projection_flat[selected_indices] = grad_flat[selected_indices]

            proj_norm = projection_flat.norm() + eps
            if proj_norm <= eps:
                projection_flat = grad_flat.clone()
            else:
                projection_flat = projection_flat * (total_norm / proj_norm)
            projection_flat = projection_flat * target_sign

            direction_flat = projection_flat
            direction_norm = direction_flat.norm() + eps
            actual_cosine_sim = torch.dot(direction_flat, grad_flat) / (direction_norm * total_norm)

            # 累积统计信息
            total_cosine_sim += actual_cosine_sim.item()
            total_rank += len(selected_indices)

            direction = []
            start = 0
            for g in bp_grads:
                sz = g.numel()
                dir_tensor = direction_flat[start:start+sz].view_as(g)
                if g.device != device:
                    dir_tensor = dir_tensor.to(g.device)
                direction.append(dir_tensor)
                start += sz

            # 在最后一次迭代时打印汇总（放在 yield 之前，避免消费者停止后汇总丢失）
            if idx == q - 1:
                total_elapsed = time.time() - total_start_time
                avg_cosine_sim = total_cosine_sim / q
                avg_rank = total_rank / q
                print(
                    f"[Block Instruct] Summary: avg_rank={avg_rank:.1f}, avg_cosine_similarity={avg_cosine_sim:.6f}, "
                    f"total_time={total_elapsed:.4f}s, device={device}",
                    flush=True,
                )

            yield direction

    return generator()

def generate_instruct_directions(bp_grads, q, cosine_target, total_norm, total_norm_sq):
    """生成与BP梯度方向保持给定余弦相似度的方向迭代器。"""
    if not bp_grads or q is None or q <= 0:
        return None

    if total_norm <= 1e-12:
        return None

    cosine_target = float(cosine_target)
    cosine_target = max(min(cosine_target, 0.9999), -0.9999)
    orth_scale = math.sqrt(max(0.0, 1.0 - cosine_target ** 2))
    eps = 1e-12

    def generator():
        for idx in range(q):
            noises = None
            for _attempt in range(6):
                noises = [torch.randn_like(g) for g in bp_grads]
                dot = 0.0
                for noise, grad in zip(noises, bp_grads):
                    dot += float(torch.sum(noise * grad).item())
                proj_coeff = dot / (total_norm_sq + eps)
                for i in range(len(noises)):
                    noises[i] = noises[i] - proj_coeff * bp_grads[i]

                noise_norm_sq = 0.0
                for noise in noises:
                    noise_norm_sq += float(torch.sum(noise * noise).item())

                if noise_norm_sq > eps:
                    noise_norm = math.sqrt(noise_norm_sq)
                    for i in range(len(noises)):
                        noises[i] = noises[i] / (noise_norm + eps)
                    break
            else:
                noises = [torch.randn_like(g) for g in bp_grads]
                noise_norm_sq = sum(float(torch.sum(n * n).item()) for n in noises) + eps
                noise_norm = math.sqrt(noise_norm_sq)
                for i in range(len(noises)):
                    noises[i] = noises[i] / (noise_norm + eps)

            direction = []
            for grad, noise in zip(bp_grads, noises):
                dir_tensor = cosine_target * grad + orth_scale * total_norm * noise
                direction.append(dir_tensor)
            
            # # 计算实际的余弦相似度并打印日志
            # dot_product = 0.0
            # direction_norm_sq = 0.0
            # for dir_t, grad in zip(direction, bp_grads):
            #     dot_product += float(torch.sum(dir_t * grad).item())
            #     direction_norm_sq += float(torch.sum(dir_t * dir_t).item())
            
            # direction_norm = math.sqrt(direction_norm_sq + eps)
            # actual_cosine_sim = dot_product / (direction_norm * total_norm + eps)
            
            # print(f"[Instruct] Perturbation {idx+1}/{q}: cosine_similarity={actual_cosine_sim:.6f}, target={cosine_target:.6f}")
            
            yield direction

    return generator()


# --- 3.5. 优化器实现 (Optimizer Implementations) ---

def _zeropower_via_newtonschulz5_torch(G, steps=5):
    """Newton-Schulz 迭代计算零次幂投影（用于 MuDaMW）"""
    assert G.dim() == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G
    if G.shape[0] > G.shape[1]:
        X = X.T
    X = X / (torch.norm(X) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X

def _adjust_lr_for_muon_torch(lr, shape):
    """调整学习率（用于 MuDaMW）"""
    A, B = shape[:2]
    return lr * (0.2 * math.sqrt(max(A, B)))

class CustomAdamOptimizer:
    """自定义 Adam 优化器（可用于 FO 和 ZO）"""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self, grads=None):
        """
        执行一步优化更新
        grads: 可选，外部提供的梯度列表（用于 ZO）。如果为 None，则使用参数的 .grad
        """
        self.t += 1
        
        for i, p in enumerate(self.params):
            if grads is not None:
                g = grads[i]
            else:
                if p.grad is None:
                    continue
                g = p.grad.data
            
            if g is None:
                continue
            
            # Adam 更新
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

    def state_dict(self):
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            't': self.t,
            'm': [m.clone() for m in self.m],
            'v': [v.clone() for v in self.v],
        }

    def load_state_dict(self, state):
        self.lr = state.get('lr', self.lr)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.epsilon = state.get('epsilon', self.epsilon)
        self.t = state.get('t', 0)

        m_state = state.get('m', [])
        v_state = state.get('v', [])
        if len(m_state) != len(self.m) or len(v_state) != len(self.v):
            raise ValueError("State dict does not match parameter groups for CustomAdamOptimizer")

        for i in range(len(self.m)):
            self.m[i] = m_state[i].clone().to(device=self.params[i].device, dtype=self.params[i].dtype)
            self.v[i] = v_state[i].clone().to(device=self.params[i].device, dtype=self.params[i].dtype)

class MuDaMWOptimizer:
    """MuDaMW 优化器（基于 flwr_server.py 的实现）"""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-6, 
                 weight_decay=0.0, correct_bias=True, cautious=False, hidden_size=768):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.correct_bias = correct_bias
        self.cautious = cautious
        self.hidden_size = hidden_size
        self.t = 0
        self.exp_avg = [torch.zeros_like(p.data) for p in self.params]
        self.exp_avg_sq = [torch.zeros_like(p.data) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self, grads=None):
        """
        执行一步优化更新
        grads: 可选，外部提供的梯度列表（用于 ZO）。如果为 None，则使用参数的 .grad
        """
        self.t += 1
        
        for i, p in enumerate(self.params):
            if grads is not None:
                g = grads[i]
            else:
                if p.grad is None:
                    continue
                g = p.grad.data
            
            if g is None:
                continue
            
            # Decoupled weight decay
            if self.weight_decay > 0.0:
                p.data.add_(p.data, alpha=-self.lr * self.weight_decay)
            
            # 一二阶动量
            self.exp_avg[i] = self.beta1 * self.exp_avg[i] + (1.0 - self.beta1) * g
            self.exp_avg_sq[i] = self.beta2 * self.exp_avg_sq[i] + (1.0 - self.beta2) * (g ** 2)
            
            denom = torch.sqrt(self.exp_avg_sq[i]) + self.epsilon
            
            # 偏置校正
            step_size = self.lr
            if self.correct_bias:
                bias_correction1 = 1.0 - (self.beta1 ** self.t)
                bias_correction2 = 1.0 - (self.beta2 ** self.t)
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
            
            # 归一化梯度（可选 cautious）
            if self.cautious:
                mask = (self.exp_avg[i] * g > 0).float()
                scale = mask.numel() / (mask.sum() + 1.0)
                mask = mask * scale
                norm_grad = (self.exp_avg[i] * mask) / denom
            else:
                norm_grad = self.exp_avg[i] / denom
            
            # 一维向量 → 2D 正交化
            if norm_grad.dim() == 1 and norm_grad.numel() % self.hidden_size == 0 and norm_grad.numel() > self.hidden_size:
                G = norm_grad.reshape(self.hidden_size, -1)
                adj_lr = _adjust_lr_for_muon_torch(step_size, G.shape)
                G = _zeropower_via_newtonschulz5_torch(G, steps=5)
                norm_grad = G.reshape(-1)
                step = adj_lr
            elif norm_grad.dim() == 2:
                # 对于 2D 参数（如权重矩阵），直接应用正交化
                adj_lr = _adjust_lr_for_muon_torch(step_size, norm_grad.shape)
                norm_grad = _zeropower_via_newtonschulz5_torch(norm_grad, steps=5)
                step = adj_lr
            else:
                step = step_size
            
            p.data.add_(norm_grad, alpha=-step)

    def state_dict(self):
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'correct_bias': self.correct_bias,
            'cautious': self.cautious,
            'hidden_size': self.hidden_size,
            't': self.t,
            'exp_avg': [buf.clone() for buf in self.exp_avg],
            'exp_avg_sq': [buf.clone() for buf in self.exp_avg_sq],
        }

    def load_state_dict(self, state):
        self.lr = state.get('lr', self.lr)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.epsilon = state.get('epsilon', self.epsilon)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        self.correct_bias = state.get('correct_bias', self.correct_bias)
        self.cautious = state.get('cautious', self.cautious)
        self.hidden_size = state.get('hidden_size', self.hidden_size)
        self.t = state.get('t', 0)

        exp_avg_state = state.get('exp_avg', [])
        exp_avg_sq_state = state.get('exp_avg_sq', [])
        if len(exp_avg_state) != len(self.exp_avg) or len(exp_avg_sq_state) != len(self.exp_avg_sq):
            raise ValueError("State dict does not match parameter groups for MuDaMWOptimizer")

        for i in range(len(self.exp_avg)):
            self.exp_avg[i] = exp_avg_state[i].clone().to(device=self.params[i].device, dtype=self.params[i].dtype)
            self.exp_avg_sq[i] = exp_avg_sq_state[i].clone().to(device=self.params[i].device, dtype=self.params[i].dtype)

# --- 4. 训练循环 (Training Loops) ---

def train(
    mode,
    scope,
    q,
    lr,
    epochs,
    batch_size,
    device,
    plot_file,
    csv_file=None,
    log_interval=10,
    optimizer_type='sgd',
    bp_interval=None,
    queries_use_different_data=False,
    model_size='200M',
    dataset_name='cosmopedia-100k',
    max_samples=None,
    checkpoint_dir=None,
    logger=None,
    run_name=None,
    bp_dataset_name=None,
    bp_max_samples=None,
    blend_bp_gradient=False,
):
    """主训练函数"""
    
    # 设置
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = create_model(model_size=model_size, vocab_size=len(tokenizer)).to(device)
    dataloader = get_dataloader(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    
    # 为BP创建单独的dataloader（如果指定了不同的数据集）
    bp_dataloader = None
    if bp_dataset_name is not None and bp_dataset_name != dataset_name:
        print(f"Creating separate BP dataloader with dataset: {bp_dataset_name}")
        bp_dataloader = get_dataloader(
            tokenizer=tokenizer,
            dataset_name=bp_dataset_name,
            batch_size=batch_size,
            max_samples=bp_max_samples,
        )
        if logger:
            logger.info("Separate BP dataloader created with dataset: %s", bp_dataset_name)

    csv_path = Path(csv_file) if csv_file else None
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(
            "Starting training run '%s' with configuration: mode=%s scope=%s q=%s lr=%s epochs=%s batch_size=%s optimizer=%s bp_interval=%s blend_bp_gradient=%s device=%s dataset=%s model_size=%s max_samples=%s bp_dataset=%s bp_max_samples=%s",
            run_name or "unnamed",
            mode,
            scope,
            q,
            lr,
            epochs,
            batch_size,
            optimizer_type,
            bp_interval,
            blend_bp_gradient,
            device,
            dataset_name,
            model_size,
            max_samples,
            bp_dataset_name or "same_as_main",
            bp_max_samples or "default",
        )
    
    # 确定可训练参数
    trainable_params = get_trainable_parameters(model, scope)
    params_trainable = sum(p.numel() for p in trainable_params)
    params_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {params_trainable / 1e6:.2f}M ({params_trainable*100/params_total:.2f}% of total)")
    if logger:
        logger.info(
            "Trainable parameters: %.2fM (%.2f%% of total %.2fM)",
            params_trainable / 1e6,
            params_trainable * 100 / params_total,
            params_total / 1e6,
        )

    # 冻结非训练参数
    for p in model.parameters():
        p.requires_grad = False
    for p in trainable_params:
        p.requires_grad = True

    zo_like_modes = {'ZO', 'Calibrate', 'Instruct'}

    query_batch_provider = None
    if queries_use_different_data and mode in zo_like_modes:
        query_dataloader = DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=True)
        query_iter = iter(query_dataloader)

        def _next_query_batch():
            nonlocal query_iter
            try:
                batch = next(query_iter)
            except StopIteration:
                query_iter = iter(query_dataloader)
                batch = next(query_iter)
            batch = batch.to(device)
            labels = batch.clone()
            return batch, labels

        query_batch_provider = _next_query_batch
        print("ZO queries will use fresh data batches per direction.")
        if logger:
            logger.info("ZO queries will use fresh data batches per direction.")

    # 为BP创建单独的batch provider（如果使用单独的数据集）
    bp_batch_provider = None
    if bp_dataloader is not None:
        bp_iter = iter(bp_dataloader)

        def _next_bp_batch():
            nonlocal bp_iter
            try:
                batch = next(bp_iter)
            except StopIteration:
                bp_iter = iter(bp_dataloader)
                batch = next(bp_iter)
            batch = batch.to(device)
            labels = batch.clone()
            return batch, labels

        bp_batch_provider = _next_bp_batch
        print(f"BP will use separate dataset: {bp_dataset_name}")
        if logger:
            logger.info("BP will use separate dataset: %s", bp_dataset_name)

    if mode in {'Calibrate', 'Instruct'}:
        if bp_interval is None or bp_interval <= 0:
            raise ValueError(f"Mode '{mode}' requires bp_interval > 0.")

    # 初始化优化器和损失函数
    optimizer = None
    if mode == 'FO':
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(trainable_params, lr=lr)
        elif optimizer_type == 'adam':
            optimizer = CustomAdamOptimizer(trainable_params, lr=lr)
        elif optimizer_type == 'mudamw':
            optimizer = MuDaMWOptimizer(trainable_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        print(f"Using optimizer: {optimizer_type.upper()}")
        if logger:
            logger.info("Using optimizer: %s", optimizer_type.upper())
    elif mode in zo_like_modes:
        if optimizer_type == 'sgd':
            optimizer = None  # 使用 vanilla SGD（手动更新）
            print(f"Using optimizer: Vanilla SGD (manual update)")
            if logger:
                logger.info("Using optimizer: Vanilla SGD (manual update)")
        elif optimizer_type == 'adam':
            optimizer = CustomAdamOptimizer(trainable_params, lr=lr)
            print(f"Using optimizer: Custom Adam")
            if logger:
                logger.info("Using optimizer: Custom Adam")
        elif optimizer_type == 'mudamw':
            optimizer = MuDaMWOptimizer(trainable_params, lr=lr)
            print(f"Using optimizer: MuDaMW")
            if logger:
                logger.info("Using optimizer: MuDaMW")
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    loss_fn = CrossEntropyLoss()
    
    losses = []
    
    # 初始化CSV日志
    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'epoch',
                'step',
                'mode',
                'scope',
                'q',
                'lr',
                'batch_size',
                'optimizer',
                'bp_interval',
                'loss',
                'grad_norm'
            ])
    
    # 开始训练
    model.train()
    step = 0
    last_metrics = {
        'loss': None,
        'grad_norm': None,
        'epoch': None,
        'step': None,
    }
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        if logger:
            logger.info("Epoch %s/%s started", epoch + 1, epochs)
        for batch in pbar:
            inputs = batch.to(device)
            labels = inputs.clone()

            grad_norm = 0.0  # 默认值
            
            if mode == 'FO':
                if hasattr(optimizer, 'zero_grad'):
                    optimizer.zero_grad()
                logits = model(inputs).logits
                # Shift logits and labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss.backward()
                
                # 计算梯度范数（用于记录）
                grad_norm_sq = 0.0
                for p in trainable_params:
                    if p.grad is not None:
                        grad_norm_sq += float(torch.sum(p.grad.detach() * p.grad.detach()).item())
                grad_norm = math.sqrt(grad_norm_sq)
                
                optimizer.step()
            
            elif mode in zo_like_modes:
                should_use_bp = (
                    mode in {'Calibrate', 'Instruct'}
                    and bp_interval is not None
                    and bp_interval > 0
                    and ((step + 1) % bp_interval == 0)
                )

                bp_grads = None
                if should_use_bp:
                    # 如果有单独的BP数据集，则使用它；否则使用当前训练batch
                    if bp_batch_provider is not None:
                        bp_inputs, bp_labels = bp_batch_provider()
                        loss, bp_grads = compute_backprop_gradients(model, trainable_params, loss_fn, bp_inputs, bp_labels)
                    else:
                        loss, bp_grads = compute_backprop_gradients(model, trainable_params, loss_fn, inputs, labels)
                else:
                    with torch.no_grad():
                        current_logits = model(inputs).logits
                        # Shift logits and labels for next-token prediction
                        shift_logits = current_logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                epsilon = 1e-4  # 增大扰动大小以提高数值稳定性

                manual_dirs = None
                total_norm_sq = None
                total_norm = None
                if mode == 'Instruct' and should_use_bp and bp_grads is not None:
                    total_norm_sq = 0.0
                    for g in bp_grads:
                        total_norm_sq += float(torch.sum(g * g).item())
                    total_norm = math.sqrt(total_norm_sq)
                    if total_norm > 0.0:
                        manual_dirs = generate_instruct_directions_with_R(
                            bp_grads,
                            q,
                            INSTRUCT_COSINE_TARGET,
                            total_norm,
                            device=device,
                        )
                    if manual_dirs is None and total_norm is not None and total_norm > 0.0:
                        manual_dirs = ([g / total_norm for g in bp_grads],)

                grad_paramwise = zo_gradient_estimator(
                    model,
                    trainable_params,
                    loss_fn,
                    inputs,
                    labels,
                    q,
                    epsilon,
                    device,
                    manual_directions=manual_dirs,
                    data_provider=query_batch_provider,
                )

                # 在Calibrate模式下使用BP梯度
                if mode == 'Calibrate' and should_use_bp and bp_grads is not None:
                    # grad_paramwise = [
                        # 0.5 * (gz + gb)
                        # for gz, gb in zip(grad_paramwise, bp_grads)
                    # ]
                    grad_paramwise = bp_grads
                
                # 在Instruct模式下，可选择混合BP和ZO梯度
                if mode == 'Instruct' and blend_bp_gradient and should_use_bp and bp_grads is not None:
                    grad_paramwise = [
                        0.5 * (gz + gb)
                        for gz, gb in zip(grad_paramwise, bp_grads)
                    ]

                grad_norm_sq = 0.0
                for g in grad_paramwise:
                    if g is not None:
                        grad_norm_sq += float(torch.sum(g.detach() * g.detach()).item())
                grad_norm = math.sqrt(grad_norm_sq)

                if optimizer is None:
                    for p, g in zip(trainable_params, grad_paramwise):
                        if g is None:
                            continue
                        p.data -= lr * g
                else:
                    optimizer.step(grads=grad_paramwise)

            losses.append(loss.item())
            current_step = step
            last_metrics.update({
                'loss': float(loss.item()),
                'grad_norm': float(grad_norm),
                'epoch': epoch + 1,
                'step': current_step,
            })

            # 记录到CSV / 日志 / checkpoint（每 log_interval 步）
            should_log_step = (log_interval > 0) and (current_step % log_interval == 0)
            if should_log_step:
                timestamp_dt = datetime.now()
                timestamp = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
                row_q = q if mode in zo_like_modes else 'N/A'
                row_bp_interval = bp_interval if mode in {'Calibrate', 'Instruct'} else 'N/A'

                if csv_path:
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            timestamp,
                            epoch + 1,
                            current_step,
                            mode,
                            scope,
                            row_q,
                            lr,
                            batch_size,
                            optimizer_type,
                            row_bp_interval,
                            loss.item(),
                            grad_norm
                        ])

                if logger:
                    logger.info(
                        "step=%s epoch=%s loss=%.6f grad_norm=%.6f mode=%s scope=%s q=%s bp_interval=%s",
                        current_step,
                        epoch + 1,
                        loss.item(),
                        grad_norm,
                        mode,
                        scope,
                        row_q,
                        row_bp_interval,
                    )

                if checkpoint_path:
                    optimizer_state = None
                    if optimizer is not None and hasattr(optimizer, 'state_dict'):
                        optimizer_state = optimizer.state_dict()

                    metadata = {
                        'timestamp': timestamp_dt.isoformat(),
                        'run_name': run_name,
                        'mode': mode,
                        'scope': scope,
                        'epoch': epoch + 1,
                        'step': current_step,
                        'q': q if mode in zo_like_modes else None,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'optimizer': optimizer_type,
                        'bp_interval': bp_interval if mode in {'Calibrate', 'Instruct'} else None,
                        'loss': float(loss.item()),
                        'grad_norm': float(grad_norm),
                        'device': device,
                        'model_size': model_size,
                        'dataset': dataset_name,
                        'checkpoint_type': 'periodic',
                    }
                    save_latest_checkpoint(
                        model,
                        tokenizer,
                        checkpoint_path,
                        optimizer_state=optimizer_state,
                        metadata=metadata,
                        logger=logger,
                    )
            
            postfix = {
                "loss": f"{loss.item():.4f}",
                "grad_norm": f"{grad_norm:.4f}",
                "opt": optimizer_type
            }
            if mode in zo_like_modes:
                postfix["queries"] = f"{q}"
                if mode in {'Calibrate', 'Instruct'} and bp_interval is not None and bp_interval > 0:
                    postfix["bp_int"] = bp_interval

            pbar.set_postfix(postfix)
            
            step += 1
        if logger:
            logger.info("Epoch %s/%s completed", epoch + 1, epochs)

    if checkpoint_path and last_metrics['loss'] is not None:
        optimizer_state = None
        if optimizer is not None and hasattr(optimizer, 'state_dict'):
            optimizer_state = optimizer.state_dict()
        final_metadata = {
            'timestamp': datetime.now().isoformat(),
            'run_name': run_name,
            'mode': mode,
            'scope': scope,
            'epoch': last_metrics['epoch'],
            'step': last_metrics['step'],
            'q': q if mode in zo_like_modes else None,
            'learning_rate': lr,
            'batch_size': batch_size,
            'optimizer': optimizer_type,
            'bp_interval': bp_interval if mode in {'Calibrate', 'Instruct'} else None,
            'loss': last_metrics['loss'],
            'grad_norm': last_metrics['grad_norm'],
            'device': device,
            'model_size': model_size,
            'dataset': dataset_name,
            'checkpoint_type': 'final',
        }
        save_latest_checkpoint(
            model,
            tokenizer,
            checkpoint_path,
            optimizer_state=optimizer_state,
            metadata=final_metadata,
            logger=logger,
        )

    if logger:
        logger.info("Training complete. Total steps: %s", step)

    # --- 5. 结果可视化 (Result Visualization) ---
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    q_text = q if mode in zo_like_modes else 'N/A'
    bp_text = bp_interval if mode in {'Calibrate', 'Instruct'} else 'N/A'
    plt.title(
        f'Training Loss Curve\nMode={mode}, Scope={scope}, q={q_text}, BP-Interval={bp_text}, '
        f'LR={lr}, Optimizer={optimizer_type.upper()}'
    )
    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True)
    plt.savefig(plot_file)
    print(f"\nTraining finished. Loss curve saved to '{plot_file}'")


# --- 6. 主程序入口 (Main Entry Point) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce 'Zeroth Order Optimization for Pretraining Language Models' paper's vanilla solver experiment.")
    parser.add_argument("--mode", type=str, required=True, choices=['FO', 'ZO', 'Calibrate', 'Instruct'], help="Optimization mode.")
    parser.add_argument("--scope", type=str, default='reduced', choices=['full', 'reduced'], help="Training scope: 'full' model or 'reduced' (last layer only).")
    parser.add_argument("--query_budget_q", type=int, default=1, help="Query budget (q) for ZO. Number of random directions.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam', 'mudamw'], 
                        help="Optimizer type: SGD (vanilla), Adam, or MuDaMW.")
    parser.add_argument("--csv_file", type=str, default=None, help="CSV file to save training logs.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval for CSV (every N steps).")
    parser.add_argument("--bp_interval", type=int, default=0, help="Backprop interval for hybrid modes (Calibrate/Instruct). Set > 0 to enable.")
    parser.add_argument(
        "--queries_use_different_data",
        action="store_true", default=True,
        help="Use a fresh data batch for each ZO query instead of reusing the training batch.",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Directory to store run logs. Defaults to logs/<run_name>_<timestamp>.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to store the latest checkpoint. Defaults to <log_dir>/checkpoint.")
    parser.add_argument("--disable_checkpoint", action="store_true", help="Disable checkpoint saving.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name to organize logs and checkpoints.")
    
    # 模型和数据集参数
    parser.add_argument("--model_size", type=str, default='200M', 
                        choices=['20M', '200M', '500M', '1B'],
                        help="Model size: 20M (fast), 200M (GPT-2 Small), 500M (medium), 1B (large).")
    parser.add_argument("--dataset", type=str, default='cosmopedia-100k',
                        choices=['cosmopedia-100k', 'cosmopedia', 'wikitext-103', 'openwebtext', 
                                'c4', 'tinystories', 'pile-subset', 'fineweb', 'fineweb-edu', 
                                'fineweb-edu-10bt'],
                        help="Dataset name for training.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use from dataset. None=use recommended value.")
    
    # BP数据集参数（用于Calibrate/Instruct模式）
    parser.add_argument("--bp_dataset", type=str, default=None,
                        choices=['cosmopedia-100k', 'cosmopedia', 'wikitext-103', 'openwebtext', 
                                'c4', 'tinystories', 'pile-subset', 'fineweb', 'fineweb-edu', 
                                'fineweb-edu-10bt'],
                        help="Separate dataset for BP gradient computation (Calibrate/Instruct modes). If not specified, uses same as --dataset.")
    parser.add_argument("--bp_max_samples", type=int, default=None,
                        help="Maximum number of samples to use from BP dataset. None=use recommended value.")
    
    # 梯度混合参数（用于Instruct模式）
    parser.add_argument("--blend_bp_gradient", action="store_true", default=True,
                        help="In Instruct mode, blend BP gradient with ZO gradient (average of both). Only effective when bp_interval > 0.")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    run_start = datetime.now()
    timestamp_str = run_start.strftime("%Y%m%d_%H%M%S")

    q_str = args.query_budget_q if args.mode in {'ZO', 'Calibrate', 'Instruct'} else 'na'
    bp_str = args.bp_interval if args.mode in {'Calibrate', 'Instruct'} else 'na'
    default_run_name = f"{args.mode}_{args.scope}_q{q_str}_bp{bp_str}_opt{args.optimizer}_lr{args.learning_rate}_bs{args.batch_size}"
    run_name = args.run_name or default_run_name

    if args.log_dir:
        run_log_dir = Path(args.log_dir)
        if not run_log_dir.is_absolute():
            run_log_dir = Path.cwd() / run_log_dir
    else:
        base_log_dir = Path.cwd() / "logs"
        run_log_dir = base_log_dir / f"{run_name}_{timestamp_str}"

    run_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be stored in {run_log_dir}")

    log_file = run_log_dir / f"training_{timestamp_str}.log"
    logger = logging.getLogger(f"reproduce_zo_paper.{run_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.info("Run directory initialized at %s", run_log_dir)

    if args.csv_file is None:
        csv_file_path = run_log_dir / f"{run_name}.csv"
    else:
        csv_file_path = Path(args.csv_file)
        if not csv_file_path.is_absolute():
            csv_file_path = run_log_dir / csv_file_path

    if args.disable_checkpoint:
        checkpoint_path = None
    else:
        if args.checkpoint_dir:
            checkpoint_path = Path(args.checkpoint_dir)
            if not checkpoint_path.is_absolute():
                checkpoint_path = run_log_dir / args.checkpoint_dir
        else:
            checkpoint_path = run_log_dir / "checkpoint"

    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plot_filename = f"{args.mode}_{args.scope}_q{q_str}_bp{bp_str}_opt{args.optimizer}_lr{args.learning_rate}_bs{args.batch_size}.png"

    bp_interval_arg = args.bp_interval if args.bp_interval > 0 else None

    train(
        mode=args.mode,
        scope=args.scope,
        q=args.query_budget_q,
        lr=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        plot_file=results_dir / plot_filename,
        csv_file=str(csv_file_path),
        log_interval=args.log_interval,
        optimizer_type=args.optimizer,
        bp_interval=bp_interval_arg,
        queries_use_different_data=args.queries_use_different_data,
        model_size=args.model_size,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        checkpoint_dir=str(checkpoint_path) if checkpoint_path else None,
        logger=logger,
        run_name=run_name,
        bp_dataset_name=args.bp_dataset,
        bp_max_samples=args.bp_max_samples,
        blend_bp_gradient=args.blend_bp_gradient,
    )

    print(f"CSV metrics saved to {csv_file_path}")
    if checkpoint_path:
        print(f"Latest checkpoint stored at {checkpoint_path}")
        logger.info("Latest checkpoint stored at %s", checkpoint_path)
    else:
        print("Checkpoint saving disabled.")
        logger.info("Checkpoint saving disabled.")
    print(f"Training logs saved to {log_file}")
    logger.info("Training logs saved to %s", log_file)

    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
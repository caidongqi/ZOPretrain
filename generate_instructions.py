import torch
import time
from datetime import datetime

def generate_instruct_directions_with_R(bp_grads, q, cosine_target, total_norm, total_norm_sq=None, device=None):
    """生成基于低秩投影R的方向，使其与BP梯度具有较高余弦相似度。"""
    if not bp_grads or q is None or q <= 0:
        return None
    if total_norm <= 1e-12:
        return None

    if device is None:
        device = bp_grads[0].device if hasattr(bp_grads[0], 'device') else torch.device('cpu')

    cosine_target = float(cosine_target)
    cosine_target = max(min(cosine_target, 0.9999), -0.9999)
    target_sign = -1.0 if cosine_target < 0.0 else 1.0

    min_similarity = max(abs(cosine_target), 0.9)
    min_similarity = min(min_similarity, 0.9999)
    eps = 1e-12

    grad_flat = torch.cat([g.flatten().to(device) for g in bp_grads])
    d = grad_flat.numel()
    total_norm_sq = float(torch.sum(grad_flat * grad_flat).item())
    if total_norm_sq <= eps:
        return None

    energy_threshold = min_similarity ** 2 * total_norm_sq

    abs_sq = grad_flat.abs().pow(2)
    initial_rank = min(64, d)
    k = max(1, initial_rank)

    while True:
        values, indices = torch.topk(abs_sq, k, largest=True)
        captured_energy = float(values.sum().item())
        if captured_energy >= energy_threshold or k >= d:
            break
        previous_k = k
        k = min(k * 2, d)
        if k == previous_k:
            break

    cumsum_values = torch.cumsum(values, dim=0)
    if energy_threshold <= float(cumsum_values[-1].item()):
        effective_rank_idx = torch.searchsorted(cumsum_values, torch.tensor(energy_threshold, device=cumsum_values.device))
        effective_rank = int(effective_rank_idx.item()) + 1
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
                out = direction_flat[start:start+sz].view_as(g)
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
                    f"[Instruct] Summary: avg_rank={avg_rank:.1f}, avg_cosine_similarity={avg_cosine_sim:.6f}, "
                    f"total_time={total_elapsed:.4f}s",
                    flush=True,
                )

            yield direction

    return generator()


def generate_instruct_directions_hybrid(
    bp_grads,
    q,
    cosine_target,
    noise_scale,
    device=None,
):
    """Generate hybrid directions that preserve high-energy gradient components and add noise elsewhere."""
    if not bp_grads or q is None or q <= 0:
        return None

    sample_grad = bp_grads[0]
    if device is None:
        device = sample_grad.device if hasattr(sample_grad, "device") else torch.device("cpu")
    dtype = sample_grad.dtype if hasattr(sample_grad, "dtype") else torch.float32

    grad_flat = torch.cat([g.flatten().to(device=device, dtype=dtype) for g in bp_grads])
    d = int(grad_flat.numel())
    if d == 0:
        return None

    total_norm = torch.norm(grad_flat)
    if total_norm <= 1e-12:
        return None
    total_norm_sq = total_norm * total_norm

    cosine_target = float(cosine_target)
    noise_scale = float(noise_scale)
    min_similarity = max(min(abs(cosine_target), 0.9999), 0.9)
    energy_threshold = float(min_similarity ** 2 * total_norm_sq)

    abs_sq = grad_flat.abs().pow(2)
    k = min(64, d)
    while True:
        values, indices = torch.topk(abs_sq, k, largest=True)
        captured_energy = float(values.sum().item())
        if captured_energy >= energy_threshold or k >= d:
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

    def generator():
        total_cosine = 0.0
        start_time = time.time()

        for direction_idx in range(int(q)):
            hybrid_flat = torch.zeros_like(grad_flat)
            hybrid_flat[high_energy_indices] = grad_flat[high_energy_indices]

            low_energy_mask = torch.ones(grad_flat.shape, dtype=torch.bool, device=device)
            low_energy_mask[high_energy_indices] = False
            num_low_energy_dims = int(low_energy_mask.sum().item())

            if num_low_energy_dims > 0 and noise_scale > 0.0:
                noise_magnitude = noise_scale * (total_norm / (d ** 0.5))
                noise = torch.randn(num_low_energy_dims, device=device, dtype=dtype) * noise_magnitude
                hybrid_flat[low_energy_mask] = noise

            hybrid_norm = torch.norm(hybrid_flat)
            if hybrid_norm <= 1e-12:
                final_similarity = 0.0
            else:
                final_similarity = float(torch.dot(hybrid_flat, grad_flat) / (hybrid_norm * total_norm))
            total_cosine += final_similarity

            print("-" * 60)
            print(
                f"[Instruct-Hybrid] direction #{direction_idx + 1}: "
                f"d={d}, effective_rank={effective_rank} ({effective_rank / d:.2%} of total)"
            )
            print(f"  cosine_target={cosine_target}")
            print(f"  noise_scale={noise_scale}")
            print(f"  actual_cosine_similarity={final_similarity:.6f}")
            print("-" * 60)

            direction = []
            start = 0
            for grad in bp_grads:
                size = grad.numel()
                chunk = hybrid_flat[start:start + size].view_as(grad)
                if chunk.device != grad.device or chunk.dtype != grad.dtype:
                    chunk = chunk.to(device=grad.device, dtype=grad.dtype)
                direction.append(chunk)
                start += size

            if direction_idx == int(q) - 1:
                elapsed = time.time() - start_time
                average_cosine = total_cosine / max(1, int(q))
                print(
                    f"[Instruct-Hybrid] Summary: avg_cosine_similarity={average_cosine:.6f}, "
                    f"effective_rank={effective_rank}, "
                    f"total_time={elapsed:.4f}s",
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

def generate_instruct_directions(bp_grads, q, cosine_target, total_norm, total_norm_sq, device):
    """生成与BP梯度方向保持给定余弦相似度的方向迭代器。"""
    if not bp_grads or q is None or q <= 0:
        return None

    if total_norm <= 1e-12:
        return None

    cosine_target = float(cosine_target)
    cosine_target = max(min(cosine_target, 0.9999), -0.9999)
    orth_scale = math.sqrt(max(0.0, 1.0 - cosine_target ** 2))
    eps = 1e-12

    # 将 total_norm_sq 转换为 GPU tensor
    total_norm_sq_tensor = torch.tensor(total_norm_sq, device=device, dtype=torch.float32)
    total_norm_tensor = torch.tensor(total_norm, device=device, dtype=torch.float32)
    eps_tensor = torch.tensor(eps, device=device, dtype=torch.float32)

    def generator():
        total_start_time = time.time()
        total_cosine_sim = 0.0
        
        for idx in range(q):
            noises = None
            for _attempt in range(6):
                # 在 GPU 上生成随机噪声
                noises = [torch.randn_like(g, device=device) for g in bp_grads]
                
                # GPU 上计算点积
                dot = torch.tensor(0.0, device=device)
                for noise, grad in zip(noises, bp_grads):
                    dot += torch.sum(noise * grad)
                
                # GPU 上计算投影系数
                proj_coeff = dot / (total_norm_sq_tensor + eps_tensor)
                
                # GPU 上进行正交化
                for i in range(len(noises)):
                    noises[i] = noises[i] - proj_coeff * bp_grads[i]

                # GPU 上计算噪声范数
                noise_norm_sq = torch.tensor(0.0, device=device)
                for noise in noises:
                    noise_norm_sq += torch.sum(noise * noise)

                if noise_norm_sq > eps:
                    noise_norm = torch.sqrt(noise_norm_sq)
                    for i in range(len(noises)):
                        noises[i] = noises[i] / (noise_norm + eps_tensor)
                    break
            else:
                # 失败后的处理
                noises = [torch.randn_like(g, device=device) for g in bp_grads]
                noise_norm_sq = torch.tensor(0.0, device=device)
                for n in noises:
                    noise_norm_sq += torch.sum(n * n)
                noise_norm = torch.sqrt(noise_norm_sq + eps_tensor)
                for i in range(len(noises)):
                    noises[i] = noises[i] / (noise_norm + eps_tensor)

            # GPU 上生成最终方向
            direction = []
            for grad, noise in zip(bp_grads, noises):
                dir_tensor = cosine_target * grad + orth_scale * total_norm_tensor * noise
                direction.append(dir_tensor)
            
            # 计算实际的余弦相似度
            dot_product = torch.tensor(0.0, device=device)
            direction_norm_sq = torch.tensor(0.0, device=device)
            for dir_t, grad in zip(direction, bp_grads):
                dot_product += torch.sum(dir_t * grad)
                direction_norm_sq += torch.sum(dir_t * dir_t)
            
            direction_norm = torch.sqrt(direction_norm_sq + eps_tensor)
            actual_cosine_sim = dot_product / (direction_norm * total_norm_tensor + eps_tensor)
            
            # 累积统计信息
            total_cosine_sim += actual_cosine_sim.item()
            
            # 在最后一次迭代时打印汇总
            if idx == q - 1:
                total_elapsed = time.time() - total_start_time
                avg_cosine_sim = total_cosine_sim / q
                print(
                    f"[Instruct-GramSchmidt] Summary: avg_cosine_similarity={avg_cosine_sim:.6f}, "
                    f"target={cosine_target:.6f}, total_time={total_elapsed:.4f}s, device={device}",
                    flush=True,
                )
            
            yield direction

    return generator()
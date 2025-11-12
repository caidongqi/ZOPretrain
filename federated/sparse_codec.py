import base64
import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch


# -----------------------
# Varint (base-128) utils
# -----------------------

def _varint_encode_uint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint only supports non-negative integers")
    out = bytearray()
    v = int(value)
    while True:
        to_write = v & 0x7F
        v >>= 7
        if v:
            out.append(to_write | 0x80)
        else:
            out.append(to_write)
            break
    return bytes(out)


def _varint_decode_uint(data: bytes, offset: int = 0) -> Tuple[int, int]:
    shift = 0
    result = 0
    pos = offset
    while True:
        if pos >= len(data):
            raise ValueError("varint decode overflow")
        b = data[pos]
        pos += 1
        result |= ((b & 0x7F) << shift)
        if (b & 0x80) == 0:
            break
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")
    return result, pos


# -----------------------------------
# Index delta coding (sorted indices)
# -----------------------------------

def _encode_sorted_indices_varint(indices_sorted: np.ndarray) -> bytes:
    if indices_sorted.size == 0:
        # encode length=0
        return _varint_encode_uint(0)
    if indices_sorted.dtype.kind not in ("i", "u"):
        raise ValueError("indices must be integer dtype")
    # Ensure ascending and unique
    diffs = np.diff(indices_sorted)
    if np.any(diffs <= 0):
        raise ValueError("indices must be strictly increasing")
    b = bytearray()
    # first write length
    b.extend(_varint_encode_uint(int(indices_sorted.size)))
    # write first absolute index
    b.extend(_varint_encode_uint(int(indices_sorted[0])))
    # write deltas
    for delta in diffs:
        b.extend(_varint_encode_uint(int(delta)))
    return bytes(b)


def _decode_sorted_indices_varint(buf: bytes) -> np.ndarray:
    pos = 0
    length, pos = _varint_decode_uint(buf, pos)
    if length == 0:
        return np.zeros((0,), dtype=np.int64)
    first, pos = _varint_decode_uint(buf, pos)
    out = np.empty((length,), dtype=np.int64)
    out[0] = int(first)
    prev = out[0]
    for i in range(1, length):
        delta, pos = _varint_decode_uint(buf, pos)
        prev = prev + int(delta)
        out[i] = prev
    return out


# ---------------------------
# Int8 symmetric quantization
# ---------------------------

def _quantize_int8(values: np.ndarray) -> Tuple[np.ndarray, float]:
    if values.size == 0:
        return values.astype(np.int8, copy=False), 1.0
    vmax = float(np.max(np.abs(values)))
    if vmax <= 0.0 or not np.isfinite(vmax):
        scale = 1.0
        q = np.zeros_like(values, dtype=np.int8)
    else:
        scale = vmax / 127.0
        q = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
    return q, float(scale)


def _dequantize_int8(qvalues: np.ndarray, scale: float) -> np.ndarray:
    if qvalues.size == 0:
        return qvalues.astype(np.float32, copy=False)
    return (qvalues.astype(np.float32)) * float(scale)


# ---------------------------------------------
# Public API: encode/decode shared-support dirs
# ---------------------------------------------

def encode_shared_sparse_dirs_from_flats(
    dir_flats: List[torch.Tensor],
    d: int,
    shapes: List[List[int]],
    total_norm: float,
    *,
    noise_seed: int,
    noise_policy: str = "zero",   # "zero" | "all" | "low"
    alpha: float = 0.1,
    device: Optional[torch.device] = None,
) -> str:
    """
    Encode K directions sharing one support set S into a compact JSON with:
      - support indices as varint-delta bytes (base64)
      - per-direction int8 values (base64) + per-direction scale (float)
      - noise config: seed/policy/alpha
    Inputs:
      dir_flats: list of 1-D torch float tensors length d
      d: full dimensionality
      shapes: parameter shapes for reconstruction on server
      total_norm: reference norm for noise scaling
    Returns:
      JSON string (utf-8) suitable to put into metrics/config.
    """
    if len(dir_flats) == 0:
        payload: Dict[str, Any] = {
            "version": 1,
            "d": int(d),
            "shapes": shapes,
            "support_b64": base64.b64encode(_encode_sorted_indices_varint(np.zeros((0,), dtype=np.int64))).decode("ascii"),
            "K": 0,
            "values_b64": [],
            "scales": [],
            "total_norm": float(total_norm),
            "noise": {
                "seed": int(noise_seed),
                "policy": str(noise_policy),
                "alpha": float(alpha),
            },
        }
        return json.dumps(payload)

    # Prefer operate on provided device (GPU) to reduce host transfers
    dev = device if device is not None else (dir_flats[0].device if len(dir_flats) > 0 else torch.device("cpu"))
    # Build shared support mask on device
    support_mask = torch.zeros((int(d),), dtype=torch.bool, device=dev)
    for t in dir_flats:
        flat = t.detach().to(dev, dtype=torch.float32).view(-1)
        support_mask |= (flat != 0.0)
    support_idx = torch.nonzero(support_mask, as_tuple=False).view(-1)
    support_idx, _ = torch.sort(support_idx)
    support = support_idx.detach().to("cpu").numpy().astype(np.int64, copy=False)
    # Encode support
    support_bytes = _encode_sorted_indices_varint(support)
    support_b64 = base64.b64encode(support_bytes).decode("ascii")

    # Gather per-direction values on support and quantize
    values_b64: List[str] = []
    scales: List[float] = []
    if support.size == 0:
        # All zeros
        for _ in dir_flats:
            values_b64.append(base64.b64encode(b"").decode("ascii"))
            scales.append(1.0)
    else:
        idx = support_idx.to(dev)
        for t in dir_flats:
            flat = t.detach().to(dev, dtype=torch.float32).view(-1)
            vals = flat.index_select(0, idx)  # on device
            # Quantize on device
            vmax = torch.max(torch.abs(vals)) if vals.numel() > 0 else torch.tensor(0.0, device=dev)
            vmax_f = float(vmax.item())
            if not np.isfinite(vmax_f) or vmax_f <= 0.0:
                scale = 1.0
                q = torch.zeros_like(vals, dtype=torch.int8, device=dev)
            else:
                scale = vmax_f / 127.0
                q = torch.clamp(torch.round(vals / scale), -127, 127).to(torch.int8)
            values_b64.append(base64.b64encode(q.detach().to("cpu").numpy().tobytes()).decode("ascii"))
            scales.append(float(scale))

    payload = {
        "version": 1,
        "d": int(d),
        "shapes": shapes,
        "support_b64": support_b64,
        "K": int(len(dir_flats)),
        "values_b64": values_b64,
        "scales": scales,
        "total_norm": float(total_norm),
        "noise": {
            "seed": int(noise_seed),
            "policy": str(noise_policy),
            "alpha": float(alpha),
        },
    }
    return json.dumps(payload)


def decode_shared_sparse_dirs_to_numpy(
    payload_json: str,
    *,
    apply_noise: bool = True,
    device: Optional[torch.device] = None,
) -> List[List[np.ndarray]]:
    """
    Decode the shared-support sparse JSON back to per-parameter numpy arrays.
    Optionally apply noise according to the encoded noise config:
      - policy "zero": apply only to complement of support
      - policy "all": apply to all dimensions
      - policy "low": alias of "zero"
    Returns:
      List of K directions; each is a list of numpy arrays per parameter shape (float32).
    """
    blob = json.loads(payload_json)
    d = int(blob.get("d", 0))
    shapes = [tuple(x) for x in blob.get("shapes", [])]
    support_b64 = blob.get("support_b64", "")
    K = int(blob.get("K", 0))
    values_b64 = blob.get("values_b64", [])
    scales = blob.get("scales", [])
    total_norm = float(blob.get("total_norm", 0.0))
    noise = blob.get("noise", {}) or {}
    seed = int(noise.get("seed", 0))
    policy = str(noise.get("policy", "zero")).lower()
    alpha = float(noise.get("alpha", 0.0))

    support_bytes = base64.b64decode(support_b64.encode("ascii")) if support_b64 else b""
    support = _decode_sorted_indices_varint(support_bytes) if support_bytes else np.zeros((0,), dtype=np.int64)
    s = int(support.size)

    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    support_idx_t = torch.from_numpy(support.astype(np.int64, copy=False)).to(dev) if s > 0 else torch.empty((0,), dtype=torch.long, device=dev)

    # Prepare noise if needed
    apply_policy_zero = policy in ("zero", "low")
    add_noise_all = policy == "all"
    noise_std = float(alpha) * (float(total_norm) / (np.sqrt(max(d, 1))))

    # Precompute masks for noise
    if add_noise_all:
        noise_mask_t = torch.ones((d,), dtype=torch.bool, device=dev)
    elif apply_policy_zero:
        noise_mask_t = torch.ones((d,), dtype=torch.bool, device=dev)
        if s > 0:
            noise_mask_t.index_fill_(0, support_idx_t, False)
    else:
        noise_mask_t = torch.zeros((d,), dtype=torch.bool, device=dev)

    # Generator for deterministic noise on device
    gen = torch.Generator(device=dev.type).manual_seed(int(seed))

    results: List[List[np.ndarray]] = []
    for k in range(K):
        flat_t = torch.zeros((d,), dtype=torch.float32, device=dev)
        if s > 0:
            vb64 = values_b64[k] if k < len(values_b64) else ""
            qbytes = base64.b64decode(vb64.encode("ascii")) if vb64 else b""
            if qbytes:
                q_np = np.frombuffer(qbytes, dtype=np.int8).copy()
                q_t = torch.from_numpy(q_np).to(dev)
            else:
                q_t = torch.empty((0,), dtype=torch.int8, device=dev)
            scale_k = float(scales[k]) if k < len(scales) else 1.0
            vals_t = q_t.to(torch.float32) * float(scale_k)
            if vals_t.numel() != s:
                raise ValueError("decoded value length mismatch with support size")
            flat_t.index_copy_(0, support_idx_t, vals_t)
        if apply_noise and noise_std > 0.0:
            if bool(torch.any(noise_mask_t).item()):
                if add_noise_all:
                    noise = torch.randn((d,), generator=gen, device=dev, dtype=torch.float32) * float(noise_std)
                    flat_t.add_(noise)
                else:
                    idx_mask_t = torch.nonzero(noise_mask_t, as_tuple=False).view(-1)
                    if idx_mask_t.numel() > 0:
                        noise = torch.randn((idx_mask_t.numel(),), generator=gen, device=dev, dtype=torch.float32) * float(noise_std)
                        flat_t.index_add_(0, idx_mask_t, noise)
        # reshape per parameter to numpy
        per_param: List[np.ndarray] = []
        start = 0
        for shp in shapes:
            size = int(np.prod(shp)) if len(shp) > 0 else 1
            chunk = flat_t[start:start + size].detach().to("cpu").numpy().reshape(shp).astype(np.float32, copy=False)
            per_param.append(chunk)
            start += size
        results.append(per_param)
    return results


# -----------------------
# Minimal self check API
# -----------------------

def _self_check_once(d: int = 1024, k: int = 32, K: int = 3, seed: int = 123) -> None:
    rng = np.random.default_rng(seed)
    support = np.sort(rng.choice(d, size=k, replace=False).astype(np.int64))
    dirs = []
    for i in range(K):
        flat = np.zeros((d,), dtype=np.float32)
        vals = rng.standard_normal(k).astype(np.float32) * (i + 1)
        flat[support] = vals
        dirs.append(torch.from_numpy(flat.copy()))
    shapes = [[d]]
    total_norm = float(np.sqrt(np.sum(dirs[0].numpy() ** 2)))
    payload = encode_shared_sparse_dirs_from_flats(
        [t for t in dirs],
        d,
        shapes,
        total_norm,
        noise_seed=seed,
        noise_policy="zero",
        alpha=1e-2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # decode twice to test determinism (noise with same seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec1 = decode_shared_sparse_dirs_to_numpy(payload, apply_noise=True, device=dev)
    dec2 = decode_shared_sparse_dirs_to_numpy(payload, apply_noise=True, device=dev)
    assert len(dec1) == K and len(dec2) == K
    for a, b in zip(dec1, dec2):
        assert len(a) == len(b) == 1
        # same noise seed ensures identical decode
        assert np.allclose(a[0], b[0])


def run_self_tests() -> None:
    _self_check_once()
    _self_check_once(d=4096, k=0, K=0, seed=7)  # empty case
    _self_check_once(d=4096, k=1, K=1, seed=17)  # minimal support



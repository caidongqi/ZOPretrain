import os
import pickle
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm


def _cache_file_path(cache_dir: str, block_size: int, sample_count: int) -> Path:
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    return cache_dir_path / f"dataset_bs{block_size}_samples{sample_count}.pkl"


def load_or_build_examples(tokenizer, block_size: int = 128, cache_dir: str = "cache", sample_count: int = 20000) -> List[torch.Tensor]:
    """
    读取或构建样本列表（每个样本为长度为 block_size 的 LongTensor）。
    与本地训练一致，优先从缓存加载；若无缓存则从 cosmopedia-100k 流式构建后缓存。
    """
    cache_file = _cache_file_path(cache_dir, block_size, sample_count)

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            examples = pickle.load(f)
        return examples

    dataset = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train", streaming=True)
    dataset = dataset.take(sample_count)

    def tokenize_function(examples):
        text = "".join(examples["text"])
        return tokenizer(text, truncation=False)

    tokenized_ids: List[int] = []
    for example in tqdm(dataset, desc="Reading dataset"):
        tokenized_ids.extend(tokenize_function(example)["input_ids"]) 

    examples: List[torch.Tensor] = []
    for i in range(0, len(tokenized_ids) - block_size + 1, block_size):
        examples.append(torch.tensor(tokenized_ids[i:i + block_size], dtype=torch.long))

    with open(cache_file, 'wb') as f:
        pickle.dump(examples, f)

    return examples


def get_partition_indices(total_size: int, num_clients: int, client_id: int) -> Tuple[int, int]:
    """
    均匀切分样本到各个客户端，返回 [start, end) 范围。
    """
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"client_id {client_id} out of range [0, {num_clients})")

    base = total_size // num_clients
    remainder = total_size % num_clients

    # 前 remainder 个客户端多分配 1 个样本
    if client_id < remainder:
        start = client_id * (base + 1)
        end = start + (base + 1)
    else:
        start = remainder * (base + 1) + (client_id - remainder) * base
        end = start + base

    return start, min(end, total_size)


def build_client_dataloader(tokenizer, batch_size: int, block_size: int, cache_dir: str, num_clients: int, client_id: int, sample_count: int = 20000) -> DataLoader:
    """
    为指定客户端构建其本地 DataLoader（IID 均分切片）。
    """
    examples = load_or_build_examples(tokenizer, block_size=block_size, cache_dir=cache_dir, sample_count=sample_count)
    start, end = get_partition_indices(len(examples), num_clients=num_clients, client_id=client_id)
    client_examples = examples[start:end]
    if len(client_examples) == 0:
        raise RuntimeError(f"Empty partition for client {client_id} with total {len(examples)} examples and num_clients={num_clients}")
    return DataLoader(client_examples, batch_size=batch_size, shuffle=True)


